import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import datasets.samplers as samplers
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import datasets.sampler as sampler

def get_args_parser():
    parser = argparse.ArgumentParser("OV DETR Detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
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
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")

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
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
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
        "--set_cost_emb", default=0.5, type=float, help="Euclidean distance of embeddings in matching cost, for novel class"
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=3, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)

    parser.add_argument("--feature_loss_coef", default=2, type=float)
    parser.add_argument("--contrastive_loss_coef", default=2, type=float)

    parser.add_argument("--focal_gamma", default=0, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)

    parser.add_argument("--scale_by_temperature", default=False, action="store_true")

    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument("--label_map", default=False, action="store_true")
    parser.add_argument("--max_len", default=5, type=int)

    parser.add_argument(
        "--clip_feat_path",
        default="./docs/coco_feat/clip_feat_coco.pkl",
        type=str,
    )
    parser.add_argument(
        "--prior_test_path", default="./docs/coco_feat/val_clip_prior_top6.json", type=str, help="the category priors, only used for inference, adopt this offline version to save inference time"
    )
    parser.add_argument(
        "--prototype_feat_path",
        default="./docs/prototype_clip_feat_idx_key.pkl",
        type=str,
    )
    parser.add_argument(
        "--prior_novel_train_path", default="./docs/coco_feat/train_ordered_novel_cats_prior.json", type=str, help="the cat prior for novel class, used in fine-tuning stage"
    )
    parser.add_argument("--prob", default=0.5, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--lvis_path", default="./data/lvis", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=36, type=int)
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

utils.init_distributed_mode(args)
print("git:\n  {}\n".format(utils.get_sha()))

if args.frozen_weights is not None:
    assert args.masks, "Frozen training is meant for segmentation only"
print(args)

device = torch.device(args.device)

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model, criterion, postprocessors = build_model(args)
model.to(device)

model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of params:", n_parameters)

dataset_train = build_dataset(image_set="train", args=args)
dataset_val = build_dataset(image_set="val", args=args)
#print('dataset_train.cat2label', len(dataset_train.cat2label.keys()), dataset_train.cat2label)
print("dataset_len", len(dataset_train))
print("val_dataset_len", len(dataset_val))

if args.distributed:
    if args.cache_mode:
        sampler_train = samplers.NodeDistributedSampler(dataset_train)
        sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = sampler.DistributedClassAwareSampler(dataset_train, class_sample_path='./data/coco/class_sample_train.pkl')
        print("sampler_len", len(sampler_train))
        sampler_train_2 = samplers.DistributedSampler(dataset_train)
        print("sampler_train_2", len(sampler_train_2))
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        print("val_sampler_len", len(sampler_val))
else:
    print('naive sample')
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

li_b = [i for i in (sampler_train_2)]
li_a = [j for j in (sampler_train)]
test = list(set(li_a) - set(dataset_train.ids))
#print("current sampler", len(test))
test_2 = list(set(li_b) - set(dataset_train.ids))
#print("original sampler", len(test_2))

#print("current sampler", set(li_a).issubset(dataset_train.ids))
#print("original sampler", set(li_b).issubset(dataset_train.ids))
#print(set([335047, 365138, 302707]) in dataset_train.ids)

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
data_loader_val = DataLoader(
    dataset_val,
    args.batch_size,
    sampler=sampler_val,
    drop_last=False,
    collate_fn=utils.collate_fn,
    num_workers=args.num_workers,
    pin_memory=True,
)
#print(next(iter(data_loader_val)))
print(next(iter(data_loader_train)))

