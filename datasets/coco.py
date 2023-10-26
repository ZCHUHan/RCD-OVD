# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.misc import get_local_rank, get_local_size

from .torchvision_datasets import CocoDetection as TvCocoDetection


class CocoDetection(TvCocoDetection):
    SEEN_CLASSES = (
        "toilet",
        "bicycle",
        "apple",
        "train",
        "laptop",
        "carrot",
        "motorcycle",
        "oven",
        "chair",
        "mouse",
        "boat",
        "kite",
        "sheep",
        "horse",
        "sandwich",
        "clock",
        "tv",
        "backpack",
        "toaster",
        "bowl",
        "microwave",
        "bench",
        "book",
        "orange",
        "bird",
        "pizza",
        "fork",
        "frisbee",
        "bear",
        "vase",
        "toothbrush",
        "spoon",
        "giraffe",
        "handbag",
        "broccoli",
        "refrigerator",
        "remote",
        "surfboard",
        "car",
        "bed",
        "banana",
        "donut",
        "skis",
        "person",
        "truck",
        "bottle",
        "suitcase",
        "zebra",
    )
    UNSEEN_CLASSES = (
        "umbrella",
        "cow",
        "cup",
        "bus",
        "keyboard",
        "skateboard",
        "dog",
        "couch",
        "tie",
        "snowboard",
        "sink",
        "elephant",
        "cake",
        "scissors",
        "airplane",
        "cat",
        "knife",
    )

    def __init__(
        self, img_folder, ann_file, transforms, return_masks,
        cache_mode=False, local_rank=0, local_size=1, label_map=False,
        is_train=False, prior_cats_file=None,
    ):
        super(CocoDetection, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )
        self._transforms = transforms
        #self.cat_ids = self.coco.getCatIds(self.SEEN_CLASSES + self.UNSEEN_CLASSES)
        self.cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
                        34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63,
                        65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]

        #print('self.cat_ids', self.cat_ids)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        #print('cat2label', self.cat2label)
        self.cat_ids_unseen = self.coco.getCatIds(self.UNSEEN_CLASSES)
        #self.cat_ids_unseen = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]
        #print(self.cat_ids_unseen)
        self.cat_ids_seen = self.coco.getCatIds(self.SEEN_CLASSES)
        self.prepare = ConvertCocoPolysToMask(
            return_masks, self.cat2label, label_map, self.cat_ids_unseen
        )
        self.is_train = is_train

        if prior_cats_file:
            with open(prior_cats_file, 'r') as f:
                self.prior_cats = json.load(f)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        if self.prior_cats:
            target = {"image_id": image_id, "annotations": target, "prior_cats": self.prior_cats[str(image_id)]}
        else:
            target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if len(target["labels"]) == 0:
            return self[(idx + 1) % len(self)]
        else:
            return img, target
        return img, target



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, cat2label=None, label_map=False, cat_ids_unseen=None):
        self.return_masks = return_masks
        self.cat2label = cat2label
        self.label_map = label_map
        self.cat_ids_unseen = cat_ids_unseen

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        prior_cats=None
        if "prior_cats" in target.keys():
            if self.label_map:
                prior_cats = [self.cat2label[cat] for cat in target["prior_cats"]]
            else:
                prior_cats = target["prior_cats"]
            prior_cats = torch.as_tensor(prior_cats, dtype=torch.int64)

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.label_map:
            classes = [
                self.cat2label[obj["category_id"]]
                if obj["category_id"] >= 0
                else obj["category_id"]
                for obj in anno
            ]
        else:
            classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        if prior_cats is not None:
            target["prior_labels"] = prior_cats

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "ovd_ins"
    PATHS = {
        "train": (
            root / "train2017",
            root / "ov-annotations" / f"{mode}_train2017_proposal.json", # root / "zero-shot" / f"{mode}_train2017_seen_2_proposal.json" ovd_ins_train2017_proposal.json
        ),
        "val": (root / "val2017", root / "ov-annotations" / f"{mode}_val2017_all.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    is_train=True if image_set=="train" else False
    #print('is_train ?', is_train)

    prior_path = args.prior_novel_train_path if is_train else args.prior_test_path

    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
        label_map=args.label_map,
        is_train=is_train,
        prior_cats_file=prior_path,
    )
    return dataset
