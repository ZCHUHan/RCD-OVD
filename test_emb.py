
import copy
import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.cuda.amp import GradScaler, autocast

import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from datasets.data_prefetcher import data_prefetcher
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader
from models import build_model
from models.matcher import build_matcher, build_joint_matcher
import datasets.samplers as samplers
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
from torch import nn
import argparse
import json
from collections import defaultdict

from util.misc import (
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

from models.segmentation_bak import dice_loss, sigmoid_focal_loss

# all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]

def get_args_parser():
    parser = argparse.ArgumentParser("OV DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lr_backbone_names", default=["backbone.0"], type=str, nargs="+")
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--sgd", action="store_true")

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=True, action="store_false")
    parser.add_argument("--two_stage", default=True, action="store_false")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone", default="resnet50", type=str, help="Name of the convolutional backbone to use"
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale", default=2 * np.pi, type=float, help="position / size * scale"
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer"
    )
    parser.add_argument(
        "--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer"
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=300, type=int, help="Number of query slots")
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)

    # * Segmentation
    parser.add_argument(
        "--masks", action="store_true", help="Train segmentation head if the flag is provided"
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class", default=3, type=float, help="Class coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_emb", default=2, type=float, help="Euclidean distance of embeddings in matching cost, for novel class"
    )

    # * Loss coefficients

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=3, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--soft_cls_loss_coef", default=3, type=float)
    parser.add_argument("--novel_loss_coef", default=0.9, type=float)

    parser.add_argument("--feature_loss_coef", default=2, type=float)
    parser.add_argument("--contrastive_loss_coef", default=1, type=float)

    parser.add_argument("--focal_gamma", default=0.5, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)

    parser.add_argument("--attend_nontarget_negative", default=False, action="store_true")
    parser.add_argument("--scale_by_temperature", default=False, action="store_true")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument("--label_map", default=True, action="store_true")
    parser.add_argument("--max_len", default=6, type=int)

    parser.add_argument(
        "--clip_feat_path",
        default="./docs/lvis/lvis.pkl",
        type=str,
    )
    parser.add_argument(
        "--prototype_feat_path",
        default="./docs/prototype_clip_feat_idx_key.pkl",
        type=str,
    )
    parser.add_argument(
        "--prior_test_path", default="./docs/val_clip_prior_top6.json", type=str, help="the category priors, only used for inference, adopt this offline version to save inference time"
    )
    parser.add_argument(
        "--prior_novel_train_path", default="./docs/lvis/train_novel_prior.json", type=str, help="the cat prior for novel class, used in training stage"
    )

    parser.add_argument("--prob", default=0.5, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="lvis")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--lvis_path", default="./data/lvis", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=21, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument(
        "--cache_mode", default=False, action="store_true", help="whether to cache images on memory"
    )
    parser.add_argument("--amp", default=False, action="store_true")

    return parser

parser = argparse.ArgumentParser(
    "OV DETR training and evaluation script", parents=[get_args_parser()]
)
args = parser.parse_args()

device = torch.device(args.device)

model, criterion, postprocessors = build_model(args)

for n, p in model.named_parameters():
    #print(n)
    if not p.requires_grad:
       print('not requires_grad', n) # the backbone.layer1 do not require grads


matcher = build_matcher(args)
matcher, novel_matcher = matcher

joint_matcher = build_joint_matcher(args)
model.to(device)


dataset_train = build_dataset(image_set="train", args=args)
sampler_train = torch.utils.data.RandomSampler(dataset_train)
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, args.batch_size, drop_last=True
)
data_loader_train = DataLoader(
    dataset_train,
    batch_sampler=batch_sampler_train,
    collate_fn=utils.collate_fn,
    num_workers=args.num_workers,
    pin_memory=True,
)

dataset_val = build_dataset(image_set="val", args=args)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
data_loader_val = DataLoader(
    dataset_val,
    args.batch_size,
    sampler=sampler_val,
    drop_last=False,
    collate_fn=utils.collate_fn,
    num_workers=args.num_workers,
    pin_memory=True,
)
#print("one val sample", next(iter(data_loader_val)))
#print("one train sample", next(iter(data_loader_train)))

new_clip_feat = {}

for key, value in model.clip_feat.items():
    new_clip_feat[data_loader_val.dataset.cat2label[key]] = value
model.clip_feat = new_clip_feat
prefetcher = data_prefetcher(data_loader_train, device, prefetch=True)
samples, targets = prefetcher.next()
assert 1==False
#postprocessors = PostProcess()

#model.eval()
#outputs_test = model(samples)
#orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#results, topk_boxes = postprocessors(outputs_test, orig_target_sizes)

model.train()
criterion.train()

outputs = model(samples, targets)
#loss_dict = criterion(outputs, targets)

outputs_without_aux = {
    k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
}

select_id = outputs["select_id"]
print('?select_id', select_id)
#indices = new_matcher(outputs_without_aux, targets, select_id)
indices = matcher(outputs_without_aux, targets)
print('indices', indices)
nov_indices = None
if "novel_id" in outputs_without_aux:
    print('novel id', outputs_without_aux["novel_id"])
    nov_indices = novel_matcher(outputs_without_aux)
    print('nov_indices', nov_indices)
else:
    print('no loss for novel class')

joint_indices = joint_matcher(outputs_without_aux, targets)
print('joint_indices', joint_indices[0])

#assert 0==True
#masks = []
#for t in targets:
#    mask = t["labels"] == -2
#    for ind, v in enumerate(t["labels"]):
#        if v in outputs["select_id"]:
#            mask[ind] = True
#    masks.append(mask)
#num_boxes = sum(len(t["labels"][m]) for t, m in zip(targets, masks))
num_boxes = sum(len(t["labels"]) for t in targets)
num_boxes = torch.as_tensor(
    [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
)


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in
                           enumerate(indices)])  # i equals to the batch size, (prediction_idx, gt_idx)
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def loss_labels(outputs, targets, matches, num_boxes, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    num_classes = 65
    eos_coef = 0.1
    empty_weight = torch.ones(num_classes + 1)
    empty_weight[-1] = eos_coef
    assert 'pred_logits' in outputs

    src_logits = outputs['pred_logits']
    src_logits = src_logits / 0.07

    indices, tgt_ids = matches
    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t[J] for t, (_, J) in zip(tgt_ids, indices)]).to(src_logits.device)
    target_classes = torch.full(src_logits.shape[:2], num_classes,
                                dtype=torch.int64, device=src_logits.device)

    target_classes[idx] = target_classes_o
    #print('target_classes', target_classes_novel_o, target_classes_o)

    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight.to(src_logits.device))

    losses = {'loss_ce': loss_ce}

    if log:
        # TODO this should probably be a separate loss, not hacked in this one here
        losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

    return losses

def naive_loss_labels(outputs, targets, indices, num_boxes, log=True):
    num_classes = 65
    eos_coef = 0.1
    empty_weight = torch.ones(num_classes + 1)
    empty_weight[-1] = eos_coef
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    assert 'pred_logits' in outputs
    src_logits = outputs['pred_logits']

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

    target_classes = torch.full(src_logits.shape[:2], num_classes,
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight.to(src_logits.device))
    losses = {'loss_ce': loss_ce}

    if log:
        # TODO this should probably be a separate loss, not hacked in this one here
        losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    return losses

def loss_embed(outputs, targets, indices, num_boxes, novel_indices=None):
    select_id = torch.tensor(outputs["select_id"]).to(outputs["pred_embed"].device)
    print('select_id', select_id)

    clip_query = outputs["clip_query"]  # 6(categories), 512
    target_feature = []

    #target_labels = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    #print('target_labels', target_labels)

    masks = []
    for t in targets:
        mask = t["labels"] == -2
        for ind, v in enumerate(t["labels"]):
            if v in outputs["select_id"]:
                mask[ind] = True
        masks.append(mask)
    print('final mask', masks) # length equals to bs

    for t, (src_idx, tgt_idx), m in zip(targets, indices, masks):
        for c in t["labels"][tgt_idx][m[tgt_idx]]:
            print('c', c)
            index = (select_id == c).nonzero(as_tuple=False)[0]
            target_feature.append(clip_query[index])

    target_feature = torch.cat(target_feature, dim=0)  # 按顺序对应的,有可能有重复的clip feature
    print('target_feature', target_feature.size())

    masks = torch.cat(masks)
    print('cated masks', masks)
    idx = _get_src_permutation_idx(indices)  # tuple(tensor(which batch, 0...1...1), tensor(which query, 102, 19, 255))
    src_feature = outputs["pred_embed"][idx][masks]
    print('src_feature', src_feature.size())


    if novel_indices is not None:
        novel_id = outputs["novel_id"]
        novel_idx = _get_src_permutation_idx(novel_indices)
        novel_src_feature = outputs["pred_embed"][novel_idx]
        src_feature = torch.cat([src_feature, novel_src_feature], dim=0)
        print('novel_src_feature', novel_src_feature.size(), 'final_src_feature', src_feature.size())
        novel_target_feature = clip_query[-len(novel_id):, :]
        print('novel_target_feature', novel_target_feature.size())
        target_feature = torch.cat([target_feature, novel_target_feature], dim=0)
        print('novel_target_feature', novel_target_feature.size(), 'final_target_feature', target_feature.size())
        num_boxes = num_boxes + len(novel_id)

    # l2 normalize the feature
    src_feature = nn.functional.normalize(src_feature, dim=1)
    loss_feature = F.mse_loss(src_feature, target_feature, reduction="none")
    losses = {"loss_embed": loss_feature.sum() / num_boxes}
    return losses

def loss_contrastive_tt(outputs, targets, indices, num_boxes):
    scale_by_temperature = True
    attend_nontarget_negative = True
    temperature = 0.07

    b, query_num = outputs["pred_embed"].shape[:2]
    #print('b', b, 'query_num', query_num)
    normalized_feature_emb = nn.functional.normalize(outputs["pred_embed"].flatten(0, 1), dim=1)  # BS x (num_features, =r*num_queries) x hdim, ---
    normalized_clip_emb = nn.functional.normalize(outputs["clip_query"], dim=1) # r(6) x hdim, and its dim corresponds to the dim of select_ids
    # whether normalized is the same for clip_emb, since it has already been normalized in previous step
    #print('normalized_feature_emb', normalized_feature_emb.size(), 'normalized_clip_emb', normalized_clip_emb.size())

    logits = (
        torch.matmul(normalized_feature_emb, normalized_clip_emb.transpose(-1, -2)) / temperature
    ).T  # r, bs * r * num_queries

    #print('logits', logits)

    tar_pos_map = torch.zeros(logits.shape, dtype=torch.int64)
    tar_pos_index = defaultdict(list)
    for bs, (t, (src_idx, tgt_idx)) in enumerate(zip(targets, indices)):
        cats = t["labels"][tgt_idx].tolist()
        #print('cats', cats)
        src_idx = src_idx.tolist()
        for i, (this_cat, this_src_idx) in enumerate(zip(cats, src_idx)):
            if this_cat not in tar_pos_index.keys():
                tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
            else:
                tar_pos_index[this_cat].append(bs * query_num + this_src_idx)


    select_cat_id2clip_id = {cat_id: clip_id for clip_id, cat_id in enumerate(outputs["select_id"])}
    #print('select_cat_id2clip_id', select_cat_id2clip_id)
    #print('tar_pos_index', tar_pos_index)

    index = [(torch.as_tensor([select_cat_id2clip_id[cat]], dtype=torch.int64),
              torch.as_tensor(idx, dtype=torch.int64)) for cat, idx in tar_pos_index.items()]
    #print('index', len(index), index)

    for i in index:
        tar_pos_map[i] = 1
    #print(pro_pos_map.nonzero())

    # use contrastive loss
    if not attend_nontarget_negative:
        tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
        nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
        denominator_logits = logits
    # specifically attend to 'unmatching' of nontarget-negative pairs
    else:
        negative_labels = list(set(outputs["select_id"])-set(tar_pos_index.keys()))
        # case of no negative conditions
        if len(negative_labels)==0:
            print('do not have negative conditions')
            nontar_neg_map = torch.zeros(logits.shape, dtype=torch.int64)
        else:
            #print('negative_labels', negative_labels)
            tar_neg_map = torch.zeros(logits.shape, dtype=torch.int64)
            for idx in tar_pos_index.values():
                for cat in negative_labels:
                    tar_neg_map[select_cat_id2clip_id[cat]][idx] = 1
            #print('tar_neg_index', tar_neg_map.nonzero())

            nontar_pos_map = copy.deepcopy(tar_pos_map)
            negative_idx = torch.tensor([select_cat_id2clip_id[label] for label in negative_labels])
            nontar_pos_map[negative_idx] = 1
            nontar_pos_map = ~torch.as_tensor(nontar_pos_map, dtype=torch.bool)
            nontar_pos_map = torch.as_tensor(nontar_pos_map, dtype=torch.int64)
            #print(len(nontar_pos_map.nonzero())+len(tar_pos_map.nonzero()))

            nontar_neg_map = 1-(nontar_pos_map + tar_neg_map + tar_pos_map)

        #####
        #####
        tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
        nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
        nontar_neg_map = torch.as_tensor(nontar_neg_map, dtype=torch.bool).to(logits.device)
        denominator_logits = logits.masked_fill(nontar_neg_map, -1000000)

    nominator_term = nominator_logits.sum(1)
    denominator_term = denominator_logits.logsumexp(1)
    #print('nominator_term', nominator_term)
    #print('denominator_term', denominator_term.size(), denominator_term) # (r, 1)

    num_positives_per_cat = tar_pos_map.sum(1)
    cats_with_pos = tar_pos_map.any(1)

    # nominator should be normalized by num_of_instances per category, but not for denominator
    loss = (nominator_term[num_positives_per_cat > 0] / num_positives_per_cat[
        num_positives_per_cat > 0]).sum() + denominator_term[num_positives_per_cat > 0].sum()
    #print('loss', loss)
    # clipfeature_to_proposal_loss = ((nominator_term / num_positives_per_cat + denominator_term)).masked_fill(~cats_with_pos, 0).sum()

    losses = {"loss_contrastive": loss if not scale_by_temperature else loss*temperature}
    return losses

def new_loss_contrastive(outputs, targets, indices, num_boxes):
    scale_by_temperature = True
    attend_nontarget_negative = False
    temperature = 0.07

    b, query_num = outputs["pred_embed"].shape[:2]
    #print('b', b, 'query_num', query_num)
    normalized_feature_emb = nn.functional.normalize(outputs["pred_embed"].flatten(0, 1), dim=1)  # (BS * num_features) x hdim, ---
    normalized_clip_emb = nn.functional.normalize(outputs["clip_query"], dim=1) # r(12) x hdim, and its dim corresponds to the dim of select_ids
    # whether normalized is the same for clip_emb, since it has already been normalized in previous step
    #print('normalized_feature_emb', normalized_feature_emb.size(), 'normalized_clip_emb', normalized_clip_emb.size())

    logits = (
        torch.matmul(normalized_feature_emb, normalized_clip_emb.transpose(-1, -2)) / temperature
    ).T  # r, bs * num_queries

    #print('logits', logits)

    tar_pos_map = torch.zeros(logits.shape, dtype=torch.int64)
    tar_pos_index = defaultdict(list)

    #print('select_id', outputs["select_id"])
    masks = []
    for t in targets:
        mask = t["labels"] == -2
        for ind, v in enumerate(t["labels"]):
            if v in outputs["select_id"]:
                mask[ind] = True
        masks.append(mask)
    #print('final mask', masks) # length equals to bs
    for bs, (t, (src_idx, tgt_idx), m) in enumerate(zip(targets, indices, masks)):
        #print('original cats in one batch', t["labels"])
        included_label = t["labels"][m]
        #print('included_label', included_label)
        #print('tgt_idx length', tgt_idx, len(tgt_idx))
        #print('ori_m', m, 'm', m[tgt_idx])
        ori_cats = t["labels"][tgt_idx].tolist()
        cats = t["labels"][tgt_idx][m[tgt_idx]].tolist()
        #print('ori_cats', ori_cats, 'cats', cats)
        # cats = t["labels"][tgt_idx].tolist() # in case when in selecting phase, the number already exceeds the max_len, and some annotation are not included in select_id

        src_idx = src_idx.tolist()
        for i, (this_cat, this_src_idx) in enumerate(zip(cats, src_idx)):
            if this_cat not in tar_pos_index.keys():
                tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
            else:
                tar_pos_index[this_cat].append(bs * query_num + this_src_idx)


    select_cat_id2clip_id = {cat_id: clip_id for clip_id, cat_id in enumerate(outputs["select_id"])}
    #print('select_cat_id2clip_id', select_cat_id2clip_id)
    #print('tar_pos_index', tar_pos_index)

    index = [(torch.as_tensor([select_cat_id2clip_id[cat]], dtype=torch.int64),
              torch.as_tensor(idx, dtype=torch.int64)) for cat, idx in tar_pos_index.items()]
    #print('index', len(index), index)

    for i in index:
        tar_pos_map[i] = 1
    #print(pro_pos_map.nonzero())

    # use contrastive loss
    if not attend_nontarget_negative:
        tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
        nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
        denominator_logits = logits
    # specifically attend to 'unmatching' of nontarget-negative pairs
    else:
        negative_labels = list(set(outputs["select_id"])-set(tar_pos_index.keys()))
        # case of no negative conditions
        if len(negative_labels)==0:
            print('do not have negative conditions')
            nontar_neg_map = torch.zeros(logits.shape, dtype=torch.int64)
        else:
            #print('negative_labels', negative_labels)
            tar_neg_map = torch.zeros(logits.shape, dtype=torch.int64)
            for idx in tar_pos_index.values():
                for cat in negative_labels:
                    tar_neg_map[select_cat_id2clip_id[cat]][idx] = 1
            #print('tar_neg_index', tar_neg_map.nonzero())

            nontar_pos_map = copy.deepcopy(tar_pos_map)
            negative_idx = torch.tensor([select_cat_id2clip_id[label] for label in negative_labels])
            nontar_pos_map[negative_idx] = 1
            nontar_pos_map = ~torch.as_tensor(nontar_pos_map, dtype=torch.bool)
            nontar_pos_map = torch.as_tensor(nontar_pos_map, dtype=torch.int64)
            #print(len(nontar_pos_map.nonzero())+len(tar_pos_map.nonzero()))

            nontar_neg_map = 1-(nontar_pos_map + tar_neg_map + tar_pos_map)

        #####
        #####
        tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
        nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
        nontar_neg_map = torch.as_tensor(nontar_neg_map, dtype=torch.bool).to(logits.device)
        denominator_logits = logits.masked_fill(nontar_neg_map, -100000)

    nominator_term = nominator_logits.sum(1)
    denominator_term = denominator_logits.logsumexp(1)
    #print('nominator_term', nominator_term)
    #print('denominator_term', denominator_term.size(), denominator_term) # (r, 1)

    num_positives_per_cat = tar_pos_map.sum(1)
    cats_with_pos = tar_pos_map.any(1)

    # nominator should be normalized by num_of_instances per category, but not for denominator
    loss = (nominator_term[num_positives_per_cat > 0] / num_positives_per_cat[
        num_positives_per_cat > 0]).sum() + denominator_term[num_positives_per_cat > 0].sum()
    #print('loss', loss)
    # clipfeature_to_proposal_loss = ((nominator_term / num_positives_per_cat + denominator_term)).masked_fill(~cats_with_pos, 0).sum()

    losses = {"loss_contrastive": loss if not scale_by_temperature else loss*temperature}
    return losses

def loss_contrastive_novel(outputs, indices, num_boxes):
    scale_by_temperature = True
    temperature = 0.07

    bs, query_num = outputs["pred_embed"].shape[:2]
    #print('b', b, 'query_num', query_num)
    normalized_feature_emb = nn.functional.normalize(outputs["pred_embed"].flatten(0, 1), dim=1)  # (BS * num_features) x hdim, ---
    normalized_clip_emb = nn.functional.normalize(outputs["clip_query"], dim=1) # r(12) x hdim, and its dim corresponds to the dim of select_ids

    novel_id = outputs["novel_id"]
    normalized_novel_clip_emb = normalized_clip_emb[-len(novel_id):, :]
    # whether normalized is the same for clip_emb, since it has already been normalized in previous step
    #print('normalized_feature_emb', normalized_feature_emb.size(), 'normalized_clip_emb', normalized_clip_emb.size())

    logits = (
        torch.matmul(normalized_feature_emb, normalized_novel_clip_emb.transpose(-1, -2)) / temperature
    ).T  # len(novel_id), bs * num_queries

    #print('logits', logits.size())

    tar_pos_map = torch.zeros(logits.shape, dtype=torch.int64)
    tar_pos_index = defaultdict(list)

    pseudo_targets = []
    for batch in range(bs):
        elements = novel_id[batch::bs]
        pseudo_targets.append(torch.as_tensor(elements))
    #print(pseudo_targets)
    for bs, (t, (src_idx, tgt_idx)) in enumerate(zip(pseudo_targets, indices)):

        pseudo_cats = t[tgt_idx].tolist()
        print('bs', bs, 'pseudo_cats', pseudo_cats)

        src_idx = src_idx.tolist()
        for i, (this_cat, this_src_idx) in enumerate(zip(pseudo_cats, src_idx)):
            if this_cat not in tar_pos_index.keys():
                tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
            else:
                tar_pos_index[this_cat].append(bs * query_num + this_src_idx)


    novel_cat_id2idx = {novel_cat_id: idx for idx, novel_cat_id in enumerate(novel_id)}
    index = [(torch.as_tensor([novel_cat_id2idx[cat]], dtype=torch.int64),
              torch.as_tensor(idx, dtype=torch.int64)) for cat, idx in tar_pos_index.items()]

    for i in index:
        tar_pos_map[i] = 1
    #print(pro_pos_map.nonzero())

    # use contrastive loss
    tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
    nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
    denominator_logits = logits
    # specifically attend to 'unmatching' of nontarget-negative pairs

    nominator_term = nominator_logits.sum(1)
    denominator_term = denominator_logits.logsumexp(1)
    #print('nominator_term', nominator_term)
    #print('denominator_term', denominator_term.size(), denominator_term) # (r, 1)

    num_positives_per_cat = tar_pos_map.sum(1)
    cats_with_pos = tar_pos_map.any(1)

    # nominator should be normalized by num_of_instances per category, but not for denominator
    loss = (nominator_term[num_positives_per_cat > 0] / num_positives_per_cat[
        num_positives_per_cat > 0]).sum() + denominator_term[num_positives_per_cat > 0].sum()
    #print('loss', loss)
    # clipfeature_to_proposal_loss = ((nominator_term / num_positives_per_cat + denominator_term)).masked_fill(~cats_with_pos, 0).sum()

    losses = {"loss_contrastive_novel": loss if not scale_by_temperature else loss*temperature}
    return losses

def loss_novel_labels(outputs, indices, num_boxes, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    num_classes = 65
    eos_coef = 0.1
    empty_weight = torch.ones(num_classes + 1)
    empty_weight[-1] = eos_coef
    assert 'pred_logits' in outputs
    bs, num_queries = outputs["pred_logits"].shape[:2]

    src_logits = outputs['pred_logits']
    novel_id = outputs["novel_id"]

    idx = _get_src_permutation_idx(indices)
    print('idx', idx)

    pseudo_targets = []
    for batch in range(bs):
        elements = novel_id[batch::bs]
        pseudo_targets.append(torch.as_tensor(elements))

    #print(pseudo_targets)

    target_classes_o = torch.cat([t[J] for t, (_, J) in zip(pseudo_targets, indices)]).to(src_logits.device)
    target_classes = torch.full(src_logits.shape[:2], num_classes,
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight.to(src_logits.device))

    losses = {'loss_ce_novel': loss_ce}

    if log:
        # TODO this should probably be a separate loss, not hacked in this one here
        losses['novel_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

    #print('this turn ce_loss', losses)
    return losses

def loss_labels_joint(outputs, targets, indices, num_boxes, log=True, novel_indices=None):
    num_classes = 65
    eos_coef = 0.2
    empty_weight = torch.ones(num_classes + 1)
    empty_weight[-1] = eos_coef
    gamma = 0.5
    assert 'pred_logits' in outputs
    src_logits = outputs['pred_logits']
    bs, num_queries = outputs["pred_logits"].shape[:2]

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], num_classes,
                                dtype=torch.int64, device=src_logits.device)
    # for novel class
    if novel_indices is not None:
        novel_id = outputs["novel_id"]
        novel_idx = _get_src_permutation_idx(novel_indices)
        pseudo_targets = []
        for batch in range(bs):
            elements = novel_id[batch::bs]
            #print('elements', elements)
            pseudo_targets.append(torch.as_tensor(elements))

        target_classes_novel_o = torch.cat([t[J] for t, (_, J) in zip(pseudo_targets, novel_indices)]).to(
            src_logits.device)
        #print('target_classes_novel_o', target_classes_novel_o)
        target_classes[novel_idx] = target_classes_novel_o

    # this part is latter to novel because: once the indices coincide(though very tiny likelihood), the gt can override the pseudo supervision
    target_classes[idx] = target_classes_o

    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=empty_weight.to(src_logits.device)) # bs, num_queries
#    loss_ce_bak = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=empty_weight.to(src_logits.device))

#    probs = F.softmax(src_logits, dim=-1) # bs, num_queries, c
#    pt = probs.gather(-1, target_classes.unsqueeze(-1)).squeeze(-1) # get prob of target class
#    loss_focal = loss_ce * ((1 - pt) ** gamma)

#    bg_weight = torch.ones(src_logits.shape[:2], device=src_logits.device)
#    bg_weight[target_classes == num_classes] = eos_coef
#    loss = (loss_focal * bg_weight)

    losses = {'loss_ce': loss_ce}

    if log:
        # TODO this should probably be a separate loss, not hacked in this one here
        losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

    # print('this turn ce_loss', losses)
    return losses

def loss_contrastive_joint(outputs, targets, indices, num_boxes, novel_indices=None):
    scale_by_temperature = True
    temperature = 0.07

    bs, query_num = outputs["pred_embed"].shape[:2]
    #print('b', b, 'query_num', query_num)
    normalized_feature_emb = nn.functional.normalize(outputs["pred_embed"].flatten(0, 1), dim=1)  # (BS * num_features) x hdim, ---
    normalized_clip_emb = nn.functional.normalize(outputs["clip_query"], dim=1) # r(12) x hdim, and its dim corresponds to the dim of select_ids

    logits = (
        torch.matmul(normalized_feature_emb, normalized_clip_emb.transpose(-1, -2)) / temperature
    ).T  # r, bs * num_queries

    #print('logits', logits)

    tar_pos_map = torch.zeros(logits.shape, dtype=torch.int64)
    tar_pos_index = defaultdict(list)

    if novel_indices is not None:
        novel_id = outputs["novel_id"]
        pseudo_targets = []
        for batch in range(bs):
            elements = novel_id[batch::bs]
            pseudo_targets.append(torch.as_tensor(elements))
        #print(pseudo_targets)
        for bs, (t, (src_idx, tgt_idx)) in enumerate(zip(pseudo_targets, novel_indices)):

            pseudo_cats = t[tgt_idx].tolist()
            src_idx = src_idx.tolist()
            #print('pseudo_cats', pseudo_cats, 'src_idx', src_idx)
            for i, (this_cat, this_src_idx) in enumerate(zip(pseudo_cats, src_idx)):
                if this_cat not in tar_pos_index.keys():
                    tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
                else:
                    tar_pos_index[this_cat].append(bs * query_num + this_src_idx)


    #print('select_id', outputs["select_id"])
    masks = []
    for t in targets:
        mask = t["labels"] == -2
        for ind, v in enumerate(t["labels"]):
            if v in outputs["select_id"]:
                mask[ind] = True
        masks.append(mask)
    #print('final mask', masks) # length equals to bs
    for bs, (t, (src_idx, tgt_idx), m) in enumerate(zip(targets, indices, masks)):
        cats = t["labels"][tgt_idx][m[tgt_idx]].tolist()
        src_idx = src_idx.tolist()
        print('base_cats', cats, 'base_src_idx', src_idx)
        for i, (this_cat, this_src_idx) in enumerate(zip(cats, src_idx)):
            if this_cat not in tar_pos_index.keys():
                tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
            else:
                tar_pos_index[this_cat].append(bs * query_num + this_src_idx)

    print('tar_pos_index:', tar_pos_index)
    select_cat_id2idx = {select_cat_id: idx for idx, select_cat_id in enumerate(outputs["select_id"])}
    print('select_cat_id2idx', select_cat_id2idx)
    index = [(torch.as_tensor([select_cat_id2idx[cat]], dtype=torch.int64),
              torch.as_tensor(idx, dtype=torch.int64)) for cat, idx in tar_pos_index.items()]

    for i in index:
        tar_pos_map[i] = 1
    print(tar_pos_map.nonzero())

    # use contrastive loss
    tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
    nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
    denominator_logits = logits.masked_fill(tar_pos_map, -1000000)
    # specifically attend to 'unmatching' of nontarget-negative pairs

    nominator_term = nominator_logits.sum(1)
    denominator_term = denominator_logits.logsumexp(1)

    num_positives_per_cat = tar_pos_map.sum(1)
    cats_with_pos = tar_pos_map.any(1)

    # nominator should be normalized by num_of_instances per category, but not for denominator
    loss = (nominator_term[num_positives_per_cat > 0] / num_positives_per_cat[
        num_positives_per_cat > 0]).sum() + denominator_term[num_positives_per_cat > 0].sum()

    losses = {"loss_contrastive": loss if not scale_by_temperature else loss*temperature}
    return losses

def loss_contrastive(outputs, targets, matches, num_boxes):
    scale_by_temperature=True
    temperature=0.07

    bs, query_num = outputs["pred_embed"].shape[:2]
    indices, tgt_ids = matches

    normalized_feature_emb = nn.functional.normalize(outputs["pred_embed"].flatten(0, 1),
                                                     dim=1)  # (BS * num_features) x hdim, ---
    normalized_clip_emb = nn.functional.normalize(outputs["img_feat"],
                                                  dim=1)  # r(6)*300 x hdim

    logits = (
            torch.matmul(normalized_feature_emb, normalized_clip_emb.transpose(-1, -2)) / temperature
    ).T  # len(select_id)*300, bs * num_queries
    #print('logits', logits.size())

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

    for bs, (t, (src_idx, tgt_idx), m) in enumerate(zip(tgt_ids, indices, masks)):
        cats = t[tgt_idx][m[tgt_idx]].tolist()
        src_idx = src_idx[m].tolist()

        for i, (this_cat, this_src_idx) in enumerate(zip(cats, src_idx)):
            if this_cat not in tar_pos_index.keys():
                tar_pos_index[this_cat] = [bs * query_num + this_src_idx]
            else:
                tar_pos_index[this_cat].append(bs * query_num + this_src_idx)

    select_cat_id2idx = {select_cat_id: idx for idx, select_cat_id in enumerate(outputs["select_id"])}
    index = [(torch.as_tensor([select_cat_id2idx[cat]*300], dtype=torch.int64),
              torch.as_tensor(idx, dtype=torch.int64)) for cat, idx in tar_pos_index.items()]
    #print('index', index)
    for i in index:
        tar_pos_map[i] = 1
    #print('tar_pos_index', tar_pos_index, 'select_cat_id2idx', select_cat_id2idx)
    #print(tar_pos_map.nonzero())

    # use contrastive loss
    tar_pos_map = torch.as_tensor(tar_pos_map, dtype=torch.bool).to(logits.device)
    nominator_logits = -logits.masked_fill(~tar_pos_map, 0)
    denominator_logits = logits   #.masked_fill(tar_pos_map, -1000000)


    # asymmetric, cat_to_query part
    nominator_term = nominator_logits.sum(1)
    denominator_term = denominator_logits.logsumexp(1)

    num_positives_per_cat = tar_pos_map.sum(1)

    # nominator should be normalized by num_of_instances per category, but not for denominator
    loss_cat_2_q = (nominator_term[num_positives_per_cat > 0] / num_positives_per_cat[
        num_positives_per_cat > 0]).sum() + denominator_term[num_positives_per_cat > 0].sum()

    # query_to_cat part
    nominator_term_T = nominator_logits.sum(0)
    denominator_term_T = denominator_logits.logsumexp(0)

    num_positive_query = tar_pos_map.sum(0)
    loss_q_to_cat = (nominator_term_T[num_positive_query > 0] / num_positive_query[
        num_positive_query > 0]).sum() + denominator_term_T[num_positive_query > 0].sum()

    total_loss = (loss_cat_2_q + loss_q_to_cat) / 2

    losses = {"loss_contrastive": total_loss if not scale_by_temperature else total_loss * temperature}
    return losses

def _get_box_indices(indices, novel_id):

    bs = len(indices)
    nov_sizes = torch.zeros(bs, dtype=torch.int64)
    for i in range(len(novel_id)):
        nov_sizes[i % bs] += 1

    box_indices = []
    for b, (src, tgt) in enumerate(indices):
        keep_num = len(tgt)-nov_sizes[b]
        tgt, index = torch.topk(tgt, k=keep_num, largest=False)
        src = src[index]

        box_indices.append((src, tgt))

    return box_indices

def loss_boxes(outputs, targets, matches, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
    """
    assert 'pred_boxes' in outputs
    indices, tgt_ids = matches
    # we need the exclued the novel indices, since the dataset do not contain the boxes supervisions

    if "novel_id" in outputs:
        indices = _get_box_indices(indices, outputs["novel_id"])

    idx = _get_src_permutation_idx(indices)

    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    return losses

joint_ce_loss = loss_labels(outputs, targets, joint_indices, num_boxes)
print('ce_loss', joint_ce_loss)

joint_contrastive_loss = loss_contrastive(outputs, targets, joint_indices, num_boxes)
print('contrastive_loss', joint_contrastive_loss)

joint_box_loss = loss_boxes(outputs, targets, joint_indices, num_boxes)
print('box_l1_loss', joint_box_loss)
#emd_loss = loss_embed(outputs, targets, indices, num_boxes, nov_indices)

#print('emd_loss', emd_loss)
#contrastive_loss = new_loss_contrastive(outputs, targets, indices, num_boxes)
#print('contrastive_loss', contrastive_loss)
#print('emb_loss', emd_loss, 'contrastive_loss', contrastive_loss)
#if nov_indices is not None:
#    novel_contrastive_loss = loss_contrastive_novel(outputs, nov_indices, num_boxes)
#    print('novel_contrastive_loss', novel_contrastive_loss)
#    novel_labels_loss = loss_labels_joint(outputs, nov_indices, num_boxes)
#    print('novel_labels_loss', novel_labels_loss)

#joint_loss=loss_labels_joint(outputs, targets, indices, num_boxes, novel_indices=nov_indices)
#print('joint_loss', joint_loss)
#joint_contrastive_loss = loss_contrastive_joint(outputs, targets, indices, num_boxes, novel_indices=nov_indices)
#print('joint_contrastive', joint_contrastive_loss)
'''

coco_evaluator = CocoEvaluator(get_coco_api_from_dataset(dataset_val), 'bbox', data_loader_val.dataset.cat2label, label_map=True)

coco_evaluator = torch.load('./output/evaluate.pth')
#print(len(coco_evaluator.gt_id2img_id))
#print(len(coco_evaluator.coco_eval['bbox'].evalImgs))
for iou_type, coco_eval in coco_evaluator.coco_eval.items():
    coco_eval.get_good_predict_data(coco_evaluator.gt_imgid_cat_id, coco_evaluator.gt_id2img_id)


scores = coco_eval['scores']
for idx in range(scores.shape[-3]):
    # area range index 0: all area ranges
    # max dets index -1: typically 100 per image
    score = scores[0, :, idx, 0, -1]  # all recalls given T, A, M and specific category
    score = score[score > -1]  # -1 for the precision of absent category, filter out them

    if score.size:
        average_score = np.mean(score)
        print(f"average_score of cat{all_ids[idx]}: {average_score}")
        # print('max score', score[0])


coco_ious=torch.load('./output/ious.pth')
print(len(coco_ious))

for k, v in coco_ious.items():
    if not len(v)==0:
        print(k, v.shape, v)
        v = torch.tensor(v)
        idx=torch.nonzero(v, as_tuple=True)
        print(idx)
        print(v[idx])
'''
'''
coco_val = './data/coco/ov-annotations/ovd_ins_val2017_all.json'
with open(coco_val, 'r') as f:
    coco_val = json.load(f)

print(coco_val.keys())
print(len(coco_val["annotations"]))
'''
'''
gt_ann = './data/coco/annotations/instances_train2017_seen_2_proposal.json'


#coco1 = COCO(gt_ann)
#print(len(coco1.imgs), len(coco1.cats))
clip_feat = torch.load('./docs/clip_feat_coco.pkl')
#print("clip feature size", clip_feat[32].size()) # [4776,512] [5263, 512]
for k, v in clip_feat.items():
    print('cat:', k, v.size())

all_catids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
                   36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73,
                   74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]  # noqa
catid2idx = {id: i for i, id in enumerate(all_catids)}
print(len(catid2idx.values()), catid2idx.values())'''