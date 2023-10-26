# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from collections import defaultdict

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 3, cost_bbox: float = 5, cost_giou: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
            #out_prob = (out_prob - 1) / 2 # (-1, 0) # num_queries, 66
            out_prob = (out_prob/0.07).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            #out_embed = outputs["pred_embed"].flatten(0, 1)  # [batch_size * num_queries, 512]
            #normalized_out_embed = nn.functional.normalize(out_embed, dim=1)

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]
            #print('cost_class', cost_class)

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            #print('cost_bbox', cost_bbox)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            #print('cost_giou', cost_giou)
            #print('ordinary match')

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]


            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class NOV_HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 3, cost_emb: float = 0.5, prototype_feat_path=None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_emb = cost_emb

#        self.clip_prototyep_feat = torch.load(prototype_feat_path)
        assert cost_class != 0 or cost_emb != 0 , "all costs cant be 0"

    def forward(self, outputs):
        with torch.no_grad():
            assert "novel_id" in outputs.keys()
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
            #out_prob = (out_prob - 1) / 2
            #print('novel_out_prob', out_prob.size(), out_prob)
            out_prob = (out_prob / 0.07).sigmoid()
            out_embed = outputs["pred_embed"].flatten(0, 1)  # [batch_size * num_queries, 512]
            normalized_out_embed = nn.functional.normalize(out_embed, dim=1)

            # the select_id, which contains ground-truth cats as well as pseudo novel categories inferred by clip
            novel_id = outputs["novel_id"]
            novel_clip_embed = outputs["clip_query"][-len(novel_id):, :] # novel_num, 512
            # since the clip_embed would vary in different forward select, but when measuring the distance, we use the mean embed as "prototype" embed
            # prototype_novel_embed = torch.cat([self.clip_prototyep_feat[nov] for nov in novel_id]).to(normalized_out_embed.device) # novel_num, 512
            # print('prototype_novel_embed', prototype_novel_embed.size(), prototype_novel_embed)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, novel_id]
            #print('cost class', cost_class.size(), cost_class)
            # Compute the classification cost.

            # Compute the euclidean distance
            #cost_euc_emb = torch.cdist(normalized_out_embed, novel_clip_embed, p=2)
            #print('l1 cost_euc_emb', cost_euc_emb.size(), cost_euc_emb)
            #mean_euc_emb = torch.mean(cost_euc_emb, dim=0, keepdim=True)
            #print('mean_euc_emb', mean_euc_emb.size())
            #cost_emb = torch.exp(cost_euc_emb - mean_euc_emb)
            #print('exp cost_euc_emb', cost_emb.size(), cost_emb)
            emb_sim = torch.matmul(normalized_out_embed, novel_clip_embed.transpose(-1, -2))
            cost_emb = -(emb_sim/0.07).sigmoid()
            #print("cost_emb", cost_emb.size(), cost_emb)

            #print('cost_emb', cost_emb) #'cost_emb_sim', cost_emb_sim,
            #print('novel matcher')

            # Final cost matrix
            C = self.cost_emb * cost_emb + self.cost_class * cost_class
            #cost_class should be small!! otherwise the same emb would be repeated selected
            #C = cost_class
            C = C.view(bs, num_queries, -1).cpu()

            sizes = torch.zeros(bs, dtype=torch.int64)
            for i in range(len(novel_id)):
                sizes[i%bs]+=1
            sizes = sizes[sizes.nonzero(as_tuple=True)].tolist()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            #C_cls = cost_class.view(bs, num_queries, -1).cpu()
            #indices_cls = [linear_sum_assignment(c[i]) for i, c in enumerate(C_cls.split(sizes, -1))]
            #print('indices_cls', indices_cls)

            #C_emb = cost_emb.view(bs, num_queries, -1).cpu()
            #indices_emb = [linear_sum_assignment(c[i]) for i, c in enumerate(C_emb.split(sizes, -1))]
            #print('indices_emb', indices_emb)

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class joint_matcher(nn.Module):
    def __init__(self, cost_class: float = 3, cost_bbox: float = 5, cost_giou: float = 2, cost_emb: float = 0.5,
                 novel_match=True, use_emb_cost=False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        self.cost_emb = cost_emb
        self.novel_match = novel_match
        self.use_emb_cost = use_emb_cost
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_emb != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():

            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
            out_prob = (out_prob - 1) / 2  # (-1, 0) # num_queries, 66
            # out_prob = out_prob.sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            sizes = [len(v["boxes"]) for v in targets]

            if self.novel_match and ("novel_id" in outputs):
                novel_id = outputs["novel_id"]

                # sizes part
                nov_sizes = torch.zeros(bs, dtype=torch.int64)
                for i in range(len(novel_id)):
                    nov_sizes[i % bs] += 1

                # must add corresponds to the batch order
                ids = [(b_tgt_ids.tolist()+novel_id[i::bs]) for i, b_tgt_ids in enumerate(tgt_ids.split(sizes))]
                tgt_ids = torch.cat([torch.as_tensor(item, dtype=torch.int64) for item in ids])

                # to ensure box cost for novel is large, which does not exist and causes dimension problems otherwise
                padding_boxes = torch.full((bs * num_queries, 1), 10000).to(cost_bbox)
                cost_bbox = torch.cat([torch.cat([tgt_bbox, padding_boxes.expand(-1, nov_sizes[i])], dim=-1) for i, tgt_bbox in enumerate(cost_bbox.split(sizes, dim=-1))], dim=-1)
                cost_giou = torch.cat([torch.cat([tgt_bbox, padding_boxes.expand(-1, nov_sizes[i])], dim=-1) for i, tgt_bbox in enumerate(cost_giou.split(sizes, dim=-1))], dim=-1)

                # embed part
                out_embed = outputs["pred_embed"].flatten(0, 1)  # [batch_size * num_queries, 512
                clip_embed = outputs["clip_query"]  # 6, 512
                ids2index = {cat_id: i for i, cat_id in enumerate(outputs["select_id"])} # do not need to worry the "mask" problem, since 'if' condition
                index = [ids2index[t.item()] for t in tgt_ids]
                src_embed = clip_embed[index] # 16, 512
                normalized_out_embed = nn.functional.normalize(out_embed, dim=1)

                # Compute the euclidean distance
                cost_emb = torch.cdist(normalized_out_embed, src_embed, p=2)
                #emb_sim = torch.matmul(normalized_out_embed, src_embed.transpose(-1, -2))
                #cost_emb = -(emb_sim-1) / 2

                sizes = [sizes[i] + nov_sizes[i] for i in range(len(sizes))]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            cost_class = -out_prob[:, tgt_ids]

            # Final cost matrix
            if self.use_emb_cost and ("novel_id" in outputs):
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_emb * cost_emb
            else:
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], tgt_ids.split(sizes)

class HungarianMatcher_PR(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 3, cost_bbox: float = 5, cost_giou: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
            out_prob = (out_prob - 1) / 2 # (-1, 0) # num_queries, 66
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]
            #print('cost_class', cost_class)

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            #print('cost_bbox', cost_bbox)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            #print('cost_giou', cost_giou)
            #print('ordinary match')

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            # for novel class, assign a random candidate's index (at this time, we only care about that there exist a 'concept' in image, but not care about the position)
            if "novel_id" in outputs.keys():
                novel_id = outputs["novel_id"]
                select_id = outputs["select_id"]
                novel_concept = select_id - novel_id

            #print('final_indices', indices)


            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_ori_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

def build_nov_matcher(args):
    return NOV_HungarianMatcher(cost_class=args.set_cost_class, cost_emb=args.set_cost_emb)

def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    ), NOV_HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_emb=args.set_cost_emb,
        prototype_feat_path=args.prototype_feat_path,
    )

def build_joint_matcher(args):
    return joint_matcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_emb=args.set_cost_emb,
    )