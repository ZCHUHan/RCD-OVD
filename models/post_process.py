import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops


class PostProcess_deformable(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"][..., :-1], outputs["pred_boxes"] # bs, 100, 65
        bs, num_total_queries = outputs["pred_logits"].shape[:2]
        cat_num = out_logits.shape[2]  # exclude the bg cat

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.reshape(bs, -1), 300, dim=1)
        scores = topk_values
        # print('scores', scores.size()) # 2, 300
        topk_boxes = topk_indexes // cat_num
        labels = topk_indexes % cat_num
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        # print('post process result', len(results))
        return results


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"]
        outputs_masks = F.interpolate(
            outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        outputs_masks = outputs_masks.sigmoid() > self.threshold

        for i, (cur_mask, t, tt) in enumerate(
            zip(outputs_masks, max_target_sizes, orig_target_sizes)
        ):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class OVPostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_queries=300):
        super().__init__()
        self.num_queries = num_queries

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        select_id = outputs["select_id"]
        if type(select_id) == int:
            select_id = [select_id]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid() # 2, 1800, 66

        scores, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1) # select 300 out of 65*300
        topk_boxes = topk_indexes // out_logits.shape[2]

        labels = torch.zeros_like(prob).flatten(1)
        num_queries = self.num_queries
        for ind, c in enumerate(select_id):
            labels[:, ind * num_queries : (ind + 1) * num_queries] = c # the flattened labels, 65*300
        labels = torch.gather(labels, 1, topk_boxes) # gather: 按照idx 选出元素

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results, topk_indexes

class new_PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"] # bs, 1800, 66
        bs, num_total_queries = outputs["pred_logits"].shape[:2]
        cat_num = out_logits.shape[2] - 1 # exclude the bg cat

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        #prob = out_logits.sigmoid()[..., :-1]
        prob = (out_logits + 1)/2
        prob = prob[..., :-1]
        topk_values, topk_indexes = torch.topk(prob.reshape(bs, -1), 300, dim=1)
        scores = topk_values
        # print('scores', scores.size()) # 2, 300
        topk_boxes = topk_indexes // cat_num
        labels = topk_indexes % cat_num
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        # print('post process result', len(results))
        return results


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1) # 2, 1800, 66
        scores, labels = prob[..., :-1].max(-1)
        print(prob.size(), labels.size()) #2, 1800

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results