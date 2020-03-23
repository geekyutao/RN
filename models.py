import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torchvision.models as models
from networks import G_Net, D_Net

class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class InpaintingModel(nn.Module):
    def __init__(self, g_lr, d_lr, l1_weight, gan_weight, iter=0, threshold=None):
        super(InpaintingModel, self).__init__()

        self.generator = G_Net(input_channels=3, residual_blocks=8, threshold=threshold)
        self.discriminator = D_Net(in_channels=3, use_sigmoid=True)

        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss('nsgan')

        self.g_lr, self.d_lr = g_lr, d_lr

        self.l1_weight, self.gan_weight = l1_weight, gan_weight

        self.global_iter = iter

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(self.g_lr),
            betas=(0., 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(self.d_lr),
            betas=(0., 0.9)
        )


# if __name__ == '__main__':
