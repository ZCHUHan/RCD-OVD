import torch
import copy
from clip import clip
from PIL import Image, ImageOps
from tqdm import tqdm
import json
from collections import defaultdict
from util.clip_utils import build_text_embedding_coco, build_text_embedding_lvis
from util.coco_categories import COCO_CATEGORIES

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
for _, param in model.named_parameters():
    param.requires_grad = False

# Json and COCO dataset dir path
json_path = '../data/lvis/lvis_v1_val.json'
file_dir = "../data/coco/val2017/"
save_path = "../docs/lvis/val_novel_prior.json"

with open(json_path, "r") as f:
    data = json.load(f)

img2ann_gt = defaultdict(list)
for temp in data['annotations']:
    img2ann_gt[temp['image_id']].append(temp)

text_features = build_text_embedding_lvis()
print(text_features.size())
#all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 90]  # noqa
all_ids = [*range(1, 1204, 1)]
print(len(all_ids))
idx2catid = {i: id for i, id in enumerate(all_ids)}
#assert 1==False

#NOVEL_CATS = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]
dic = {}
count=0
for image_id in tqdm(img2ann_gt.keys()):
    file_name = file_dir + f"{image_id}".zfill(12) + ".jpg"
    try:
        image = Image.open(file_name).convert("RGB")
    except FileNotFoundError:
        print("this image do not exist, skip it")
        continue
    img = preprocess(image).to(device).unsqueeze(0)
    img_features = model.encode_image(img)
    img_features /= img_features.norm(dim=-1, keepdim=True)

    #similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
    #print('similarity score', similarity.size(), similarity) # [1, 65]
    similarity_sigmoid = (img_features @ text_features.T).sigmoid()

    values, indices = similarity_sigmoid[0].topk(20)
    #scores, idxs = similarity_sigmoid[0].topk(12)
    indices = indices.tolist()
    cats = [idx2catid[index] for index in indices]
    #sorted_novel_cats = [i for i in cats if i in NOVEL_CATS]
    #print('length', len(sorted_novel_cats), sorted_novel_cats, cats)

    # Print the result
    #print('image_id', image_id)
    #print("\nTop predictions:\n")
    #for value, index in zip(values, indices):
    #    print(f"{COCO_CATEGORIES[idx2catid[index]]:>16s}: {100 * value.item():.2f}%")
    #    print('--')

    dic[image_id] = cats

with open(save_path, 'w') as f:
    json.dump(dic, f)


# torch.save(dic, save_path)

#soft_label_score = torch.load(save_path)
#with open(json_path, "r") as f:
#    data = json.load(f)


