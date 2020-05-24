#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
pwd = osp.split(osp.realpath(__file__))[0]
import sys
sys.path.append(pwd + '/..')

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn
from torchvision import transforms

import faceutils as futils
from makeup.solver_makeup import Solver_makeupGAN

cudnn.benchmark = True
solver = Solver_makeupGAN()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])


def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


def to_var(x, requires_grad=True):
    if requires_grad:
        return Variable(x).float()
    else:
        return Variable(x, requires_grad=requires_grad).float()


def copy_area(tar, src, lms):
    rect = [int(min(lms[:, 1])) - preprocess.eye_margin, 
            int(min(lms[:, 0])) - preprocess.eye_margin, 
            int(max(lms[:, 1])) + preprocess.eye_margin + 1, 
            int(max(lms[:, 0])) + preprocess.eye_margin + 1]
    tar[:, :, rect[1]:rect[3], rect[0]:rect[2]] = \
        src[:, :, rect[1]:rect[3], rect[0]:rect[2]]


def preprocess(image: Image):
    face = futils.dlib.detect(image)

    assert face, "no faces detected"

    face = face[0]
    image, face = futils.dlib.crop(image, face)

    # detect landmark
    lms = futils.dlib.landmarks(image, face) * 256 / image.width
    lms = lms.round()
    lms_eye_left = lms[42:48]
    lms_eye_right = lms[36:42]
    lms = lms.transpose((1, 0)).reshape(-1, 1, 1)   # transpose to (y-x)
    lms = np.tile(lms, (1, 256, 256))  # (136, h, w)

    # calculate relative position for each pixel
    fix = np.zeros((256, 256, 68 * 2))
    for i in range(256):  # row (y) h
        for j in range(256):  # column (x) w
            fix[i, j, :68] = i
            fix[i, j, 68:] = j
    fix = fix.transpose((2, 0, 1))  # (138, h, w)
    diff = to_var(torch.Tensor(fix - lms).unsqueeze(0), requires_grad=False)

    # obtain face parsing result
    image = image.resize((512, 512), Image.ANTIALIAS)
    mask = futils.mask.mask(image).resize((256, 256), Image.ANTIALIAS)
    mask = to_var(ToTensor(mask).unsqueeze(0), requires_grad=False)
    mask_lip = (mask == 7).float() + (mask == 9).float()
    mask_face = (mask == 1).float() + (mask == 6).float()

    mask_eyes = torch.zeros_like(mask)
    copy_area(mask_eyes, mask_face, lms_eye_left)
    copy_area(mask_eyes, mask_face, lms_eye_right)
    mask_eyes = to_var(mask_eyes, requires_grad=False)

    mask_list = [mask_lip, mask_face, mask_eyes]
    mask_aug = torch.cat(mask_list, 0)      # (3, 1, h, w)
    mask_re = F.interpolate(mask_aug, size=preprocess.diff_size).repeat(1, diff.shape[1], 1, 1)  # (3, 136, 64, 64)
    diff_re = F.interpolate(diff, size=preprocess.diff_size).repeat(3, 1, 1, 1)  # (3, 136, 64, 64)
    diff_re = diff_re * mask_re             # (3, 136, 32, 32)
    norm = torch.norm(diff_re, dim=1, keepdim=True).repeat(1, diff_re.shape[1], 1, 1)
    norm = torch.where(norm == 0, torch.tensor(1e10), norm)
    diff_re /= norm

    image = image.resize((256, 256), Image.ANTIALIAS)
    real = to_var(transform(image).unsqueeze(0))
    return [real, mask_aug, diff_re]


# parameter of eye transfer
preprocess.eye_margin = 16
# down sample size
preprocess.diff_size = (64, 64)


if __name__ == '__main__':
    # source image, reference image
    images = ['xfsy_0106.png', 'vFG586.png']
    images = [Image.open(image) for image in images]
    images = [preprocess(image) for image in images]
    transferred_image = solver.test(*(images[0]), *(images[1]))
    transferred_image.save('transferred_image.png')









