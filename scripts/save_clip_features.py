import torch
import copy
from clip import clip
from PIL import Image, ImageOps
from tqdm import tqdm
import json
from collections import defaultdict


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
for _, param in model.named_parameters():
    param.requires_grad = False
dataset = 'lvis'

# Json and COCO dataset dir path
file_dir = "./data/coco/train2017/"
if dataset == 'coco':
    json_path = '../data/coco/ov-annotations/ovd_ins_train2017_all.json'
    save_path = "../docs/clip_feat_catid_key_context.pkl"
    another_save_path = "../docs/clip_feat_idx_key_context.pkl"
    cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75,
               76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]

else:
    json_path = './data/lvis/lvis_v1_train.json'
    save_path = "../docs/lvis_feat/clip_feat_catid_key_lvis.pkl"
    another_save_path = "../docs/clip_feat_idx_key_lvis.pkl"
    cat_ids = []

with open(json_path, "r") as f:
    data = json.load(f)

img2ann_gt = defaultdict(list)
for temp in data['annotations']:
    img2ann_gt[temp['image_id']].append(temp)

dic = {}
for image_id in tqdm(img2ann_gt.keys()):
    file_name = file_dir + f"{image_id}".zfill(12) + ".jpg"
    image = Image.open(file_name).convert("RGB")
    
    for value in img2ann_gt[image_id]:
        ind = value['id']
        bbox = copy.deepcopy(value['bbox'])
        if (bbox[1] < 16) or (bbox[2] < 16):
            continue
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        try:
            roi = preprocess(image.crop(bbox)).to(device).unsqueeze(0)
        except ZeroDivisionError:
            print("this crop is bad, skip it")
            continue
        roi_features = model.encode_image(roi)
        category_id = value['category_id']

        if category_id in dic.keys():
            dic[category_id].append(roi_features) #[1, 512]
        else:
            dic[category_id] = [roi_features]

torch.save(dic, save_path)
'''
cat2idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
new_dic = {}
for key in dic.keys():
    new_dic[cat2idx[key]] = torch.cat(dic[key], 0) # [instance_num, 512]

torch.save(new_dic, another_save_path)

'''
