import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ops.histogram_matching import histogram_matching


class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def forward(self, input_data, target_data, mask_src, mask_tar):
        index_tmp = mask_src.unsqueeze(0).nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_tar.unsqueeze(0).nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]

        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        input_match = histogram_matching(
            input_masked, target_masked,
            [x_A_index, y_A_index, x_B_index, y_B_index])
        input_match = self.to_var(input_match, requires_grad=False)
        loss = F.l1_loss(input_masked, input_match)
        return loss
