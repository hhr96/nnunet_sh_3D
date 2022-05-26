import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import Function

torch.backends.cudnn.deterministic = True

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','EnergyLoss_new', 'EnergyLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

def odd_flip(H):
    '''
    generate frequency map
    when height or width of image is odd number,
    creat a array concol [0,1,...,int(H/2)+1,int(H/2),...,0]
    len(concol) = H
    '''
    m = int(H / 2)
    col = np.arange(0, m + 1)
    flipcol = col[m - 1::-1]
    concol = np.concatenate((col, flipcol), 0)
    return concol


def even_flip(H):
    '''
    generate frequency map
    when height or width of image is even number,
    creat a array concol [0,1,...,int(H/2),int(H/2),...,0]
    len(concol) = H
    '''
    m = int(H / 2)
    col = np.arange(0, m)
    flipcol = col[m::-1]
    concol = np.concatenate((col, flipcol), 0)
    return concol


def dist(target):
    '''
    sqrt(m^2 + n^2) in eq(8)
    '''

    _, _, H, W = target.shape

    if H % 2 == 1:
        concol = odd_flip(H)
    else:
        concol = even_flip(H)

    if W % 2 == 1:
        conrow = odd_flip(W)
    else:
        conrow = even_flip(W)

    m_col = concol[:, np.newaxis]
    m_row = conrow[np.newaxis, :]
    dist = np.sqrt(m_col * m_col + m_row * m_row)  # sqrt(m^2+n^2)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dist_ = torch.from_numpy(dist).float().cuda()
    else:
        dist_ = torch.from_numpy(dist).float()
    return dist_


class EnergyLoss_new(nn.Module):  # 对应unet++里的其他loss 整个loss被call是从这里开始
    def __init__(self, cuda, alpha, sigma):
        super(EnergyLoss_new, self).__init__()
        self.energylossfunc_new = EnergylossFunc_new.apply
        self.alpha = alpha
        self.cuda = cuda
        self.sigma = sigma

    def forward(self, feat, label):
        return self.energylossfunc_new(self.cuda, feat, label, self.alpha, self.sigma)


class EnergylossFunc_new(Function):
    '''
    target: ground truth
    feat: Z -0.5. Z：prob of your target class(here is vessel) with shape[B,H,W].
    Z from softmax output of unet with shape [B,C,H,W]. C: number of classes
    alpha: default 0.35
    sigma: default 0.25
    '''

    @staticmethod
    def forward(ctx, cuda, feat_levelset, target, alpha, sigma):
        hardtanh = nn.Hardtanh(min_val=0, max_val=1, inplace=False)
        target = target.float()
        index_ = dist(target) #以512乘512的图为坐标轴，计算点与点之间的距离
        dim_ = target.shape[1]
        target = torch.squeeze(target, 1)
        I1 = target + alpha * hardtanh(feat_levelset / sigma)  # G_t + alpha*H(phi) in eq(5)
        # dmn = torch.rfft(I1, s=(512, 512), dim=(2, 3, 4), norm='ortho')
        dmn = torch.fft.fft2(I1, norm='ortho') #本来应该输出[batch,512,512,2]

        dmn_r = torch.real(dmn)
        dmn_i = torch.imag(dmn)
        # dmn_r = dmn[:, :, :, 0]  # dmn's real part
        # dmn_i = dmn[:, :, :, 1]  # dmm's imagine part
        dmn2 = dmn_r * dmn_r + dmn_i * dmn_i  # dmn^2

        ctx.save_for_backward(feat_levelset, target, dmn, index_)

        F_energy = torch.sum(index_ * dmn2) / feat_levelset.shape[0] / feat_levelset.shape[1] / feat_levelset.shape[
            2]  # eq(8)

        return F_energy

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, dmn, index_ = ctx.saved_tensors
        index_ = torch.unsqueeze(index_, 0)
        # index_ = torch.unsqueeze(index_, 3)
        F_diff = 0.5 * index_ * dmn  # eq(9)
        print('in')
        diff = torch.fft.ifft2(F_diff, norm='ortho') / feature.shape[0]  # eq
        diff = torch.real(diff)
        return None, Variable(-grad_output * diff), None, None, None


class EnergyLoss(nn.Module):  # 对应unet++里的其他loss 整个loss被call是从这里开始
    def __init__(self, cuda, alpha, sigma):
        super(EnergyLoss, self).__init__()
        self.energylossfunc = EnergylossFunc.apply
        self.alpha = alpha
        self.cuda = cuda
        self.sigma = sigma

    def forward(self, feat, label):
        return self.energylossfunc(self.cuda, feat, label, self.alpha, self.sigma)


class EnergylossFunc(Function):
    '''
    target: ground truth
    feat: Z -0.5. Z：prob of your target class(here is vessel) with shape[B,H,W].
    Z from softmax output of unet with shape [B,C,H,W]. C: number of classes
    alpha: default 0.35
    sigma: default 0.25
    '''

    @staticmethod
    def forward(ctx, cuda, feat_levelset, target, alpha, sigma, Gaussian=False):
        hardtanh = nn.Hardtanh(min_val=0, max_val=1, inplace=False)
        target = target.float()
        index_ = dist(target)
        dim_ = target.shape[1]
        target = torch.squeeze(target, 1)
        I1 = target + alpha * hardtanh(feat_levelset / sigma)  # G_t + alpha*H(phi) in eq(5)
        dmn = torch.rfft(I1, 2, normalized=True, onesided=False)
        dmn_r = dmn[:, :, :, 0]  # dmn's real part
        dmn_i = dmn[:, :, :, 1]  # dmm's imagine part
        dmn2 = dmn_r * dmn_r + dmn_i * dmn_i  # dmn^2

        ctx.save_for_backward(feat_levelset, target, dmn, index_)

        F_energy = torch.sum(index_ * dmn2) / feat_levelset.shape[0] / feat_levelset.shape[1] / feat_levelset.shape[
            2]  # eq(8)

        return F_energy

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, dmn, index_ = ctx.saved_tensors
        index_ = torch.unsqueeze(index_, 0)
        index_ = torch.unsqueeze(index_, 3)
        F_diff = -0.5 * index_ * dmn  # eq(9)
        diff = torch.irfft(F_diff, 2, normalized=True, onesided=False) / feature.shape[0]  # eq
        return None, Variable(-grad_output * diff), None, None, None
