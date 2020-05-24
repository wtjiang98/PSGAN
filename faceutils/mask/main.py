#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

from .model import BiSeNet


mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
n_classes = 19
save_pth = osp.split(osp.realpath(__file__))[0] + '/resnet.pth'

net = BiSeNet(n_classes=19)
net.load_state_dict(torch.load(save_pth, map_location='cpu'))
net.eval()
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def mask(image: Image):
    assert image.size == (512, 512)
    with torch.no_grad():
        image = to_tensor(image)
        image = torch.unsqueeze(image, 0)
        out = net(image)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    for r in range(parsing.shape[0]):
        parsing[r] = [mapper[x] for x in parsing[r]]
    return Image.fromarray(parsing.astype(np.uint8))

