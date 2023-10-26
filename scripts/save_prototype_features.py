import torch
import copy
from clip import clip
from PIL import Image, ImageOps
from tqdm import tqdm
import json
from collections import defaultdict
from util.clip_utils import build_text_embedding_coco, build_text_embedding_lvis

label_map = True
text_img_ratio = 0.5
cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]
cat2idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}

clip_text_feat = build_text_embedding_coco()
# print('clip_text_feat', clip_text_feat.size()) # [65, 512]

clip_img_feat = torch.load('../docs/clip_feat_catid_key.pkl')

prototype_novel_feat = {}

for k, v in clip_img_feat.items():
    #print('cat:', k, "toidx:", cat2idx[k], v.size())
    img_embedding = v / v.norm(dim=-1, keepdim=True)
    #print('img_embedding', img_embedding.size())
    img_embedding = img_embedding.mean(dim=0) # float16 type
    #print('prototype_img_embedding', img_embedding.size(), img_embedding)

    prototype_embeding = clip_text_feat[cat2idx[k]] * text_img_ratio + img_embedding * (1 - text_img_ratio)  # cat_num, 256
    #prototype_embeding = torch.cat([clip_text_feat[cat2idx[k]], img_embedding])
    #print('prototype_embedding', prototype_embeding.size(), prototype_embeding) # float16 type
    prototype_novel_feat[cat2idx[k] if label_map else k] = torch.as_tensor(prototype_embeding, dtype=torch.float).unsqueeze(0)


save_path = '../docs/prototype_clip_feat_idx_key(conca).pkl' if label_map else "../docs/prototype_clip_feat_cat_key(conca).pkl"
#torch.save(prototype_novel_feat, save_path)

#test = torch.load('../docs/prototype_clip_feat_idx_key.pkl')
for k, v in prototype_novel_feat.items():
    print(k, v.size(), v)
