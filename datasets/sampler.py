import torch.distributed as dist

import math
import pickle as pk

import numpy as np
import torch
from torch.utils.data import BatchSampler, Sampler
from mmdet.utils import sync_random_seed


class DistributedClassAwareSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 seed=0, num_sample_class=1, class_sample_path=None):

        if num_replicas is None:
            num_replicas = dist.get_world_size() # 4
        if rank is None:
            rank = dist.get_rank() # 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = sync_random_seed(seed)

        with open(class_sample_path, "rb") as f:
            self.cat_dict = pk.load(f)
        # The number of samples taken from each per-label list
        assert num_sample_class > 0 and isinstance(num_sample_class, int)
        self.num_sample_class = num_sample_class

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # get number of images containing each category
        self.num_cat_imgs = [len(x) for x in self.cat_dict.values()]
        # filter labels without images
        self.valid_cat_inds = [
            k for k, length in zip(self.cat_dict.keys(), self.num_cat_imgs) if length != 0
        ]
        self.num_classes = len(self.valid_cat_inds)

        self.idx2localind = {idx: i for i, idx in enumerate(self.dataset.ids)}

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        # initialize label list
        label_iter_list = RandomCycleIter(self.valid_cat_inds, generator=g)
        # initialize each per-label image list
        data_iter_dict = dict()
        for i in self.valid_cat_inds:
            data_iter_dict[i] = RandomCycleIter(self.cat_dict[i], generator=g)

        def gen_cat_img_inds(cls_list, data_dict, num_sample_cls):
            """Traverse the categories and extract `num_sample_cls` image
            indexes of the corresponding categories one by one."""
            id_indices = []
            for _ in range(len(cls_list)):
                cls_idx = next(cls_list)
                for _ in range(num_sample_cls):
                    id = next(data_dict[cls_idx])
                    id_indices.append(id)
            return id_indices

        # deterministically shuffle based on epoch
        num_bins = int(
            math.ceil(self.total_size * 1.0 / self.num_classes /
                      self.num_sample_class))
        indices = []
        for i in range(num_bins):
            indices += gen_cat_img_inds(label_iter_list, data_iter_dict,
                                        self.num_sample_class)

        # fix extra samples to make it evenly divisible
        if len(indices) >= self.total_size:
            indices = indices[:self.total_size]
        else:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        final_indices = [self.idx2localind[j] for j in indices]
        return iter(final_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class RandomCycleIter:
    """Shuffle the list and do it again after the list have traversed.
    The implementation logic is referred to
    https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/datasets/loader/sampler.py
    Example:
#            >>> label_list = [0, 1, 2, 4, 5]
#            >>> g = torch.Generator()
#            >>> g.manual_seed(0)
#            >>> label_iter_list = RandomCycleIter(label_list, generator=g)
#            >>> index = next(label_iter_list)
    Args:
        data (list or ndarray): The data that needs to be shuffled.
        generator: An torch.Generator object, which is used in setting the seed
            for generating random numbers.
    """  # noqa: W605

    def __init__(self, data, generator=None):
        self.data = data
        self.length = len(data)
        self.index = torch.randperm(self.length, generator=generator).numpy()
        self.i = 0
        self.generator = generator

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.i == self.length:
            self.index = torch.randperm(
                self.length, generator=self.generator).numpy()
            self.i = 0
        idx = self.data[self.index[self.i]]
        self.i += 1
        return idx