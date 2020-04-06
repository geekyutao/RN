import torch
import torch.nn as nn
import torch.nn.functional as F

class RN_binarylabel(nn.Module):
    def __init__(self, feature_channels):
        super(RN_binarylabel, self).__init__()
        self.bn_norm = nn.BatchNorm2d(feature_channels, affine=False, track_running_stats=False)

    def forward(self, x, label):
        '''
        input:  x: (B,C,M,N), features
                label: (B,1,M,N), 1 for foreground regions, 0 for background regions
        output: _x: (B,C,M,N)
        '''
        label = label.detach()

        rn_foreground_region = self.rn(x * label, label)

        rn_background_region = self.rn(x * (1 - label), 1 - label)

        return rn_foreground_region + rn_background_region

    def rn(self, region, mask):
        '''
        input:  region: (B,C,M,N), 0 for surroundings
                mask: (B,1,M,N), 1 for target region, 0 for surroundings
        output: rn_region: (B,C,M,N)
        '''
        shape = region.size()

        sum = torch.sum(region, dim=[0,2,3])  # (B, C) -> (C)
        Sr = torch.sum(mask, dim=[0,2,3])    # (B, 1) -> (1)
        Sr[Sr==0] = 1
        mu = (sum / Sr)     # (B, C) -> (C)

        return self.bn_norm(region + (1 - mask) * mu[None,:,None,None]) * \
        (torch.sqrt(Sr / (shape[0] * shape[2] * shape[3])))[None,:,None,None]

class RN_B(nn.Module):
    def __init__(self, feature_channels):
        super(RN_B, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # RN
        self.rn = RN_binarylabel(feature_channels)    # need no external parameters

        # gamma and beta
        self.foreground_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)

    def forward(self, x, mask):
        # mask = F.adaptive_max_pool2d(mask, output_size=x.size()[2:])
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')   # after down-sampling, there can be all-zero mask

        rn_x = self.rn(x, mask)

        rn_x_foreground = (rn_x * mask) * (1 + self.foreground_gamma[None,:,None,None]) + self.foreground_beta[None,:,None,None]
        rn_x_background = (rn_x * (1 - mask)) * (1 + self.background_gamma[None,:,None,None]) + self.background_beta[None,:,None,None]

        return rn_x_foreground + rn_x_background

class SelfAware_Affine(nn.Module):
    def __init__(self, kernel_size=7):
        super(SelfAware_Affine, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gamma_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)
        self.beta_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv1(x)
        importance_map = self.sigmoid(x)

        gamma = self.gamma_conv(importance_map)
        beta = self.beta_conv(importance_map)

        return importance_map, gamma, beta

class RN_L(nn.Module):
    def __init__(self, feature_channels, threshold=0.8):
        super(RN_L, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
        return: tensor RN_L(x): (B,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # SelfAware_Affine
        self.sa = SelfAware_Affine()
        self.threshold = threshold

        # RN
        self.rn = RN_binarylabel(feature_channels)    # need no external parameters


    def forward(self, x):

        sa_map, gamma, beta = self.sa(x)     # (B,1,M,N)

        # m = sa_map.detach()
        if x.is_cuda:
            mask = torch.zeros_like(sa_map).cuda()
        else:
            mask = torch.zeros_like(sa_map)
        mask[sa_map.detach() >= self.threshold] = 1

        rn_x = self.rn(x, mask.expand(x.size()))

        rn_x = rn_x * (1 + gamma) + beta

        return rn_x
