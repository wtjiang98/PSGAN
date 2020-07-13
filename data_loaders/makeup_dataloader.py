import os
import torch
import random
import linecache

from torch.utils.data import Dataset
from PIL import Image
from tools.data_reader import DataReader
from psgan.preprocess import PreProcess


class MakeupDataloader(Dataset):
    def __init__(self, image_path, preprocess: PreProcess, transform, transform_mask):
        self.image_path = image_path
        self.transform = transform
        self.transform_mask = transform_mask
        self.preprocess = preprocess

        self.reader = DataReader(image_path)

    def __getitem__(self, index):
        (image_s, mask_s, lm_s), (image_r, mask_r, lm_r) =\
            self.reader.pick()
        lm_s = self.preprocess.relative2absolute(lm_s / image_s.size)
        lm_r = self.preprocess.relative2absolute(lm_r / image_r.size)
        image_s = self.transform(image_s)
        mask_s = self.transform_mask(Image.fromarray(mask_s))
        image_r = self.transform(image_r)
        mask_r = self.transform_mask(Image.fromarray(mask_r))

        mask_s, dist_s = self.preprocess.process(
            mask_s.unsqueeze(0), lm_s)
        mask_r, dist_r = self.preprocess.process(
            mask_r.unsqueeze(0), lm_r)
        return [image_s, mask_s, dist_s], [image_r, mask_r, dist_r]


    def __len__(self):
        return len(self.reader)
