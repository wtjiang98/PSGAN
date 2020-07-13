from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data_loaders.makeup_dataloader import MakeupDataloader
import torch
import numpy as np
import PIL
from psgan.preprocess import PreProcess

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

def get_loader(config, mode="train"):
    # return the DataLoader
    transform = transforms.Compose([
    transforms.Resize(config.DATA.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize(config.DATA.IMG_SIZE, interpolation=PIL.Image.NEAREST),
        ToTensor])

    dataset = MakeupDataloader(
        config.DATA.PATH, transform=transform,
        transform_mask=transform_mask, preprocess=PreProcess(config))
    #"""
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=True, num_workers=config.DATA.NUM_WORKERS)
    return dataloader
