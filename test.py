import torch.nn.functional as F
from torch import nn
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import lvis
import numpy as np
import json
from pathlib import Path
import pickle
import math
from collections import defaultdict
import heapq as hp
from datasets.coco_eval import CocoEvaluator, convert_to_xywh

eval = torch.load('./output/eval/latest.pth')
print(eval.keys())

predictions = eval['precision']

unseen_list = [4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63]
results_seen = []
results_unseen = []
# precision (T,R,K,A,M), T means iou(T[0] is 0.5), R means recall, K means category, A means area(all, small, medium, large), M means maxDecs()
for idx in range(predictions.shape[-3]):
    # area range index 0: all area ranges
    # max dets index -1: typically 100 per image
    prediction = predictions[0, :, idx, 0, -1]  # all recalls given T, A, M and specific category
    prediction = prediction[prediction > -1]  # -1 for the precision of absent category, filter out them
    if prediction.size:
        ap = np.mean(prediction)
        # print(f"AP {idx}: {ap}")
        if idx not in unseen_list:
            results_seen.append(float(ap * 100))
        else:
            results_unseen.append(float(ap * 100))
print(f"AP seen: {np.mean(results_seen)}")
print(f"AP unseen: {np.mean(results_unseen)}")


cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
                        34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63,
                        65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]

print(len(cat_ids))
cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
coco_eval = CocoEvaluator(coco_gt=COCO(annotation_file="./data/coco/ov-annotations/ovd_ins_val2017_all.json"), iou_types=["bbox"], cat2label=cat2label, label_map=True)
coco_eval.coco_eval['bbox'].eval = eval
coco_eval.summarize()

#print(len(coco_evaluator.img_ids))
#print(coco_evaluator.coco_eval['bbox'].eval["DTMatch"])
#print(len(coco_evaluator.gt_imgid_cat_id), coco_evaluator.gt_imgid_cat_id)
#print(len(coco_evaluator.gt_id2img_id), coco_evaluator.gt_id2img_id) # 16591
#print(len(coco_evaluator.coco_eval['bbox'].evalImgs)) # the per-image per-category evaluation results [KxAxI] elements 65*4*4836==1257360
#print(len(coco_evaluator.coco_eval['bbox']._dts.keys()), coco_evaluator.coco_eval['bbox']._dts) # 130 the last evaluated patch, useless

#print(coco_evaluator.eval_imgs['bbox'], len(coco_evaluator.eval_imgs['bbox']))
#count = 0
#for k in coco_evaluator.coco_eval['bbox'].evalImgs:
#    count+=1
#    if count>5:
#        break
#    print(k)

annotations_path = Path("./data/coco/ov-annotations/ovd_ins_train2017_proposal.json")
#with open(annotations_path, "r") as f:
#    targets_json = json.load(f)

#prior_cat_path = Path("./docs/val_clip_prior.json")
#with open(prior_cat_path, "r") as file:
#    prior_cat_json = json.load(file)
#cocoGT=COCO(annotations_path)
#print(len(cocoGT.cats))
#print('image len', len(cocoGT.imgs))
#print(cocoGT.catToImgs.keys())
#for k, v in cocoGT.catToImgs.items():
#    print('key', k, 'value', len(v))
#print(len(cocoGT.anns))
#anntations=targets_json['annotations']
#print(anntations[0]["prior_cats"], anntations[0]["category_id"], anntations[0]['bbox'], anntations[0])

#print(prior_cat_json.keys())

#for ann in targets_json['annotations']:
#    img_id = str(ann["image_id"])
#    #print(img_id, type(img_id))
#    ann['prior_cats'] = prior_cat_json[img_id]

#with open("./data/coco/ov-annotations/ovd_ins_val2017_all_2.json", "w") as wf:
#    json.dump(targets_json, wf)

save_path = "./docs/clip_prototype_feat_labelidx_key.pkl"

cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]
cat2idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
#print(cat2idx)

#all_ids = [*range(1, 1205, 1)]
#print(len(all_ids))

#clip_feat = torch.load('./docs/lvis.pkl')

#for k, v in clip_feat.items():
#    print("cat:", k, "instances size", v.size())

#print(len(clip_feat.keys()))


class_sample_path = './data/coco/class_sample_train.pkl'


#with open(class_sample_path, "wb") as f:
#    pickle.dump(cocoGT.catToImgs, f, pickle.HIGHEST_PROTOCOL)

#a = [1, 3, 5, 7, 9]
#for item in iter(a):
#    print(item)
#print('------')
#with open(class_sample_path, "rb") as f:
#    cat_dict = pickle.load(f)
#count = 0
#for k, v in cat_dict.items():
#    print('kk', k, 'vv', len(v))
#    print('vvv', len(set(v)))
#    count = count+len(set(v))
#print(count)