import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

from .segmentation_bak import dice_loss, sigmoid_focal_loss

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_gamma=0.5, focal_alpha=0.25,
                 eos_coef=0.1, temperature=0.07, scale_by_temperature=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.eos_coef = eos_coef
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def loss_labels(self, outputs, targets, matches, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        src_logits = src_logits / self.temperature

        indices, tgt_ids = matches
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(tgt_ids, indices)]).to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o
        #print('target_classes', target_classes_novel_o, target_classes_o)

        # focal scaling
        if self.focal_gamma > 0:
            loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction="none")
            probs = src_logits.sigmoid()
            pt = probs.gather(-1, target_classes.unsqueeze(-1)).squeeze(-1)  # get prob of target class
            loss_focal = loss * ((1 - pt) ** self.focal_gamma)
            bg_weight = torch.ones(src_logits.shape[:2], device=src_logits.device)
            bg_weight[target_classes == self.num_classes] = self.eos_coef
            loss_ce = (loss_focal * bg_weight).sum() / num_boxes

        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_contrastive(self, outputs, targets, matches, num_boxes):
        bs, query_num = outputs["pred_embed"].shape[:2]
        indices, tgt_ids = matches

        normalized_feature_emb = nn.functional.normalize(outputs["pred_embed"].flatten(0, 1),
                                                         dim=1)  # (BS * num_features) x hdim, ---
        normalized_clip_emb = nn.functional.normalize(outputs["clip_feat"],
                                                      dim=1)  # r(6)*300 x hdim

        logits = (
                torch.matmul(normalized_feature_emb, normalized_clip_emb.transpose(-1, -2)) / self.temperature
        ).T
        # print('logits', logits.size())

        tar_pos_map = torch.zeros(logits.shape, dtype=torch.int64)
        tar_pos_index = defaultdict(list)

        # print('select_id', outputs["select_id"])
        masks = []
        for t in tgt_ids:
            mask = t == -2
            for ind, v in enumerate(t):
                if v in outputs["select_id"]:
                    mask[ind] = True
            masks.append(mask)

        # print('final mask', masks) # length equals to bs
        for bs, (t, (src_idx, tgt_idx), m) in enumerate(zip(tgt_ids, indices, masks)):
            cats = t[tgt_idx][m[tgt_idx]].tolist()
            src_idx = src_idx[m].tolist()
            for i, (this_cat, this_src_idx) in enumerate(zip(cats, src_idx)):
                if this_cat not in tar_pos_index.keys():
                    tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
                else:
                    tar_pos_index[this_cat].append(bs * query_num + this_src_idx)

        select_cat_id2idx = {select_cat_id: idx for idx, select_cat_id in enumerate(outputs["select_id"])}
        index = [(torch.as_tensor([select_cat_id2idx[cat]*200], dtype=torch.int64),
                  torch.as_tensor(idx, dtype=torch.int64)) for cat, idx in tar_pos_index.items()]

        for i in index:
            tar_pos_map[i] = 1
        #print('tar_pos_index', tar_pos_index, 'select_cat_id2idx', select_cat_id2idx)
        #print(tar_pos_map.nonzero())

        # use contrastive loss
        tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
        nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
        denominator_logits = logits  # .masked_fill(tar_pos_map, -1000000)

        # asymmetric, cat_to_query part
        nominator_term = nominator_logits.sum(1)
        denominator_term = denominator_logits.logsumexp(1)

        num_positives_per_cat = tar_pos_map.sum(1)

        # nominator should be normalized by num_of_instances per category, but not for denominator
        loss_cat_to_q = (nominator_term[num_positives_per_cat > 0] / num_positives_per_cat[
            num_positives_per_cat > 0]).sum() + denominator_term[num_positives_per_cat > 0].sum()

        # query_to_cat part
        nominator_term_T = nominator_logits.sum(0)
        denominator_term_T = denominator_logits.logsumexp(0)

        num_positive_query = tar_pos_map.sum(0)
        loss_q_to_cat = (nominator_term_T[num_positive_query > 0] / num_positive_query[
            num_positive_query > 0]).sum() + denominator_term_T[num_positive_query > 0].sum()

        total_loss = loss_cat_to_q * 2/3 + loss_q_to_cat * 1/3

        losses = {"loss_contrastive": total_loss if not self.scale_by_temperature else total_loss * self.temperature}
        return losses

    def loss_emb(self, outputs, targets, matches, num_boxes):
        indices, tgt_ids = matches
        idx = self._get_src_permutation_idx(indices)
        src_feature = outputs["pred_embed"][idx]

        select_id = torch.tensor(outputs["select_id"]).to(src_feature.device)
        clip_query = outputs["clip_query"]
        target_feature = []
        for t, (_, i) in zip(tgt_ids, indices):
            for c in t[i]:
                #index = (select_id == c).nonzero(as_tuple=False)[0]
                index = (select_id == c).nonzero(as_tuple=False)
                target_feature.append(clip_query[index])

        target_feature = torch.cat(target_feature, dim=0)
        #print('target_feature', target_feature.size())
        # l2 normalize the feature
        src_feature = nn.functional.normalize(src_feature, dim=1)
        #print('src_feature', src_feature.size())
        loss_feature = F.mse_loss(src_feature, target_feature, reduction="none")
        losses = {"loss_embed": loss_feature.sum() / num_boxes}
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, matches, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        indices, tgt_ids = matches
        # we need the exclued the novel indices, since the dataset do not contain the boxes supervisions

        if "novel_id" in outputs:
            indices = self._get_box_indices(indices, outputs["novel_id"])

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_box_indices(self, indices, novel_id):

        bs = len(indices)
        nov_sizes = torch.zeros(bs, dtype=torch.int64)
        for i in range(len(novel_id)):
            nov_sizes[i % bs] += 1

        box_indices = []
        for b, (src, tgt) in enumerate(indices):
            keep_num = len(tgt) - nov_sizes[b]
            tgt, index = torch.topk(tgt, k=keep_num, largest=False)
            src = src[index]

            box_indices.append((src, tgt))

        return box_indices

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            #"contrastive": self.loss_contrastive,
            #"embed": self.loss_emb,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        matches = self.matcher(outputs_without_aux, targets)
        #print('indices', matches[0])

        # masks are for the case that some cats in ann but are not selected (because exceeding the max length in forward)
        masks = []
        for t in targets:
            mask = t["labels"] == -2
            for ind, v in enumerate(t["labels"]):
                if v in outputs["select_id"]:
                    mask[ind] = True
            masks.append(mask)

        num_boxes = sum(len(t["labels"][m]) for t, m in zip(targets, masks))
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, matches, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                matches = self.matcher(aux_outputs, targets)

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False

                    l_dict = self.get_loss(loss, aux_outputs, targets, matches, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            matches = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                if loss == "masks" or loss == "contrastive":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, matches, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        #print('loss', losses)
        return losses

class SetCriterion_bak(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_gamma=0.5, focal_alpha=0.25,
                 eos_coef=0.1, temperature=0.07, scale_by_temperature=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher, self.novel_matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.eos_coef = eos_coef
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def loss_labels(self, outputs, targets, indices, num_boxes, novel_indices=None, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        src_logits = src_logits / self.temperature
        bs, num_queries = outputs["pred_logits"].shape[:2]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # for novel class
#        if novel_indices is not None:
#            novel_id = outputs["novel_id"]
#            novel_idx = self._get_src_permutation_idx(novel_indices)
#            pseudo_targets = []
#            for batch in range(bs):
#                elements = novel_id[batch::bs]
#               pseudo_targets.append(torch.as_tensor(elements))

#            target_classes_novel_o = torch.cat([t[J] for t, (_, J) in zip(pseudo_targets, novel_indices)]).to(src_logits.device)
#            target_classes[novel_idx] = target_classes_novel_o

        # this part is latter to novel because: once the indices coincide(though very tiny likelihood), the gt can override the pseudo supervision
        target_classes[idx] = target_classes_o

        # focal scaling
#        if self.focal_gamma > 0:
#            loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction="none")
#            probs = src_logits.sigmoid()
#            pt = probs.gather(-1, target_classes.unsqueeze(-1)).squeeze(-1)  # get prob of target class

#            loss_focal = loss * ((1 - pt) ** self.focal_gamma)
#            bg_weight = torch.ones(src_logits.shape[:2], device=src_logits.device)
#            bg_weight[target_classes == self.num_classes] = self.eos_coef

#            loss_ce = (loss_focal * bg_weight).sum() / num_boxes

#        else:
#            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        #print('this turn ce_loss', losses)
        return losses

    def loss_contrastive(self, outputs, targets, indices, num_boxes, novel_indices=None):
        bs, query_num = outputs["pred_embed"].shape[:2]
        normalized_feature_emb = nn.functional.normalize(outputs["pred_embed"].flatten(0, 1),
                                                         dim=1)  # (BS * num_features) x hdim, ---
        normalized_clip_emb = nn.functional.normalize(outputs["clip_feat"],
                                                      dim=1)  # r(6)*300 x hdim

        logits = (
                torch.matmul(normalized_feature_emb, normalized_clip_emb.transpose(-1, -2)) / self.temperature
        ).T  # len(select_id), bs * num_queries
        # print('logits', logits)

        tar_pos_map = torch.zeros(logits.shape, dtype=torch.int64)
        tar_pos_index = defaultdict(list)

#       if novel_indices is not None:
#            novel_id = outputs["novel_id"]
#            pseudo_targets = []
#            for batch in range(bs):
#                elements = novel_id[batch::bs]
#                pseudo_targets.append(torch.as_tensor(elements))
#            # print(pseudo_targets)
#            for bs, (t, (src_idx, tgt_idx)) in enumerate(zip(pseudo_targets, novel_indices)):
#                pseudo_cats = t[tgt_idx].tolist()
#                src_idx = src_idx.tolist()
#                for i, (this_cat, this_src_idx) in enumerate(zip(pseudo_cats, src_idx)):
#                    if this_cat not in tar_pos_index.keys():
#                        tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
#                    else:
#                        tar_pos_index[this_cat].append(bs * query_num + this_src_idx)

        # print('select_id', outputs["select_id"])
        masks = []
        for t in targets:
            mask = t["labels"] == -2
            for ind, v in enumerate(t["labels"]):
                if v in outputs["select_id"]:
                    mask[ind] = True
            masks.append(mask)

        # print('final mask', masks) # length equals to bs
        for bs, (t, (src_idx, tgt_idx), m) in enumerate(zip(targets, indices, masks)):
            cats = t["labels"][tgt_idx][m[tgt_idx]].tolist()
            src_idx = src_idx[m].tolist()
            for i, (this_cat, this_src_idx) in enumerate(zip(cats, src_idx)):
                if this_cat not in tar_pos_index.keys():
                    tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
                else:
                    tar_pos_index[this_cat].append(bs * query_num + this_src_idx)

        select_cat_id2idx = {select_cat_id: idx for idx, select_cat_id in enumerate(outputs["select_id"])}
        index = [(torch.as_tensor([select_cat_id2idx[cat]*200], dtype=torch.int64),
                  torch.as_tensor(idx, dtype=torch.int64)) for cat, idx in tar_pos_index.items()]

        for i in index:
            tar_pos_map[i] = 1
        #print('tar_pos_index', tar_pos_index, 'select_cat_id2idx', select_cat_id2idx)
        #print("positive index", tar_pos_map.nonzero())

        # use contrastive loss
        tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
        nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
        denominator_logits = logits  # .masked_fill(tar_pos_map, -1000000)

        # asymmetric, cat_to_query part
        nominator_term = nominator_logits.sum(1)
        denominator_term = denominator_logits.logsumexp(1)

        num_positives_per_cat = tar_pos_map.sum(1)

        loss_cat_to_q = (nominator_term[num_positives_per_cat > 0] / num_positives_per_cat[
            num_positives_per_cat > 0]).sum() + denominator_term[num_positives_per_cat > 0].sum()

        # query_to_cat part
        nominator_term_T = nominator_logits.sum(0)
        denominator_term_T = denominator_logits.logsumexp(0)

        num_positive_query = tar_pos_map.sum(0)
        loss_q_to_cat = (nominator_term_T[num_positive_query > 0] / num_positive_query[
            num_positive_query > 0]).sum() + denominator_term_T[num_positive_query > 0].sum()

        #total_loss = loss_cat_to_q * 2/3 + loss_q_to_cat * 1/3
        total_loss = (loss_cat_to_q + loss_q_to_cat) / 2

        losses = {"loss_contrastive": total_loss if not self.scale_by_temperature else total_loss*self.temperature}
        return losses

    def loss_embed(self, outputs, targets, indices, num_boxes, novel_indices=None):
        idx = self._get_src_permutation_idx(indices) # tuple(tensor(which batch, 0...1...1), tensor(which query, 102, 19, 255))
        src_feature = outputs["pred_embed"][idx]

        select_id = torch.tensor(outputs["select_id"]).to(src_feature.device)
        clip_query = outputs["clip_query"] # 6(categories), 512

        masks = []
        for t in targets:
            mask = t["labels"] == -2
            for ind, v in enumerate(t["labels"]):
                if v in outputs["select_id"]:
                    mask[ind] = True
            masks.append(mask)

        target_feature = []
        for t, (src_idx, tgt_idx), m in zip(targets, indices, masks):
            for c in t["labels"][tgt_idx][m[tgt_idx]]:
                index = (select_id == c).nonzero(as_tuple=False)[0]
                target_feature.append(clip_query[index])

        target_feature = torch.cat(target_feature, dim=0) # 按顺序对应的,有可能有重复的clip feature
        masks = torch.cat(masks)
        src_feature = src_feature[masks]

        if novel_indices is not None:
            novel_id = outputs["novel_id"]
            novel_idx = self._get_src_permutation_idx(novel_indices)
            novel_src_feature = outputs["pred_embed"][novel_idx]
            src_feature = torch.cat([src_feature, novel_src_feature], dim=0)
            novel_target_feature = clip_query[-len(novel_id):, :]
            target_feature = torch.cat([target_feature, novel_target_feature], dim=0)
            num_boxes = num_boxes + len(novel_id)

        # l2 normalize the feature
        src_feature = nn.functional.normalize(src_feature, dim=1)
        loss_feature = F.mse_loss(src_feature, target_feature, reduction="none")
        losses = {"loss_embed": loss_feature.sum() / num_boxes}

        return losses

    def focal_loss_labels(self, outputs, targets, indices, num_boxes, novel_indices=None, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # the cosine similarity

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, #the last label is 64, so use 65 as bg label
                                    dtype=torch.int64, device=src_logits.device)

        if novel_indices is not None:
            novel_id = outputs["novel_id"]
            novel_idx = self._get_src_permutation_idx(novel_indices)
            pseudo_targets = []
            for batch in range(src_logits.shape[0]):
                elements = novel_id[batch::src_logits.shape[0]]
                pseudo_targets.append(torch.as_tensor(elements))

            target_classes_novel_o = torch.cat([t[J] for t, (_, J) in zip(pseudo_targets, novel_indices)]).to(src_logits.device)
            target_classes[novel_idx] = target_classes_novel_o

        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]], #src_logits.shape[2]+1
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits[..., :-1], target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        #print('this turn box loss', losses)
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "contrastive": self.loss_contrastive,
            #"embed": self.loss_embed,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        novel_indices = None
#        if "novel_id" in outputs_without_aux:
#            print("have novel supervision")
#            novel_indices = self.novel_matcher(outputs_without_aux)
        #print('indices', indices, 'novel_indices', novel_indices)

        # masks are for the case that some cats in ann but are not selected (because exceeding the max length in forward)
        masks = []
        for t in targets:
            mask = t["labels"] == -2
            for ind, v in enumerate(t["labels"]):
                if v in outputs["select_id"]:
                    mask[ind] = True
            masks.append(mask)

        num_boxes = sum(len(t["labels"][m]) for t, m in zip(targets, masks))
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == "labels" or loss == "embed" or loss == "contrastive":
                kwargs['novel_indices'] = novel_indices
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                novel_indices = None
#                if "novel_id" in aux_outputs:
#                    print("aux novel loss")
#                    novel_indices = self.novel_matcher(aux_outputs)

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if loss == "labels" or loss == "embed" or loss == "contrastive":
                        kwargs['novel_indices'] = novel_indices

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                if loss == "masks" or loss == "contrastive":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        #print('loss', losses)
        return losses
