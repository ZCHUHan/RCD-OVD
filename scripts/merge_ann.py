import torch
import copy

from tqdm import tqdm
import json
from collections import defaultdict
from util.coco_categories import COCO_CATEGORIES

# Json and COCO dataset dir path
json_path = '../data/coco/ov-annotations/ovd_ins_val2017_all.json'
# file_dir = "../data/coco/val2017/"
save_path = "../data/coco/ov-annotations/ovd_ins_val2017_all_with_soft_label.json"
that_path = "../docs/clip_softlabel(val_unused).pkl"

soft_label_scores = torch.load(that_path)
with open(json_path, "r") as f:
    data = json.load(f)

print(len(data['annotations']))
print(data['annotations'][0])

all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]  # noqa
catid2idx = {id: i for i, id in enumerate(all_ids)}
idx2catid = {i: id for i, id in enumerate(all_ids)}

#img2ann_gt = defaultdict(list)
for ann in tqdm(data['annotations']):
    ann['soft_label'] = soft_label_scores[ann['image_id']].squeeze().tolist()
    #img2ann_gt[ann['image_id']].append(ann)

with open(save_path, 'w') as file:
    json.dump(data, file)






