import numpy as np
import torch
import copy

def cal_hist(image):
    """
        cal cumulative hist for channel list
    """
    hists = []
    for i in range(0, 3):
        channel = image[i]
        # channel = image[i, :, :]
        channel = torch.from_numpy(channel)
        # hist, _ = np.histogram(channel, bins=256, range=(0,255))
        hist = torch.histc(channel, bins=256, min=0, max=256)
        hist = hist.numpy()
        # refHist=hist.view(256,1)
        sum = hist.sum()
        pdf = [v / sum for v in hist]
        for i in range(1, 256):
            pdf[i] = pdf[i - 1] + pdf[i]
        hists.append(pdf)
    return hists


def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table


def histogram_matching(dstImg, refImg, index):
    """
        perform histogram matching
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """
    index = [x.cpu().numpy() for x in index]
    dstImg = dstImg.detach().cpu().numpy()
    refImg = refImg.detach().cpu().numpy()
    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, 3)]
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, 3)]
    hist_ref = cal_hist(ref_align)
    hist_dst = cal_hist(dst_align)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

    mid = copy.deepcopy(dst_align)
    for i in range(0, 3):
        for k in range(0, len(index[0])):
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, 3):
        dstImg[i, index[0], index[1]] = dst_align[i]

    dstImg = torch.FloatTensor(dstImg).cuda()
    return dstImg
