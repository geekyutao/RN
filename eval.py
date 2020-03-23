from __future__ import print_function
import argparse
from math import log10
import numpy as np
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from module_util import initialize_weights
from dataset import build_dataloader
import pdb
import socket
import time
import skimage
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

from models import InpaintingModel


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Video Inpainting with Background Auxilary')
parser.add_argument('--bs', type=int, default=64, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=67454, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--img_flist', type=str, default='/data/dataset/places2/flist/val.flist')
parser.add_argument('--mask_flist', type=str, default='/data/dataset/places2/flist/3w_all.flist')
parser.add_argument('--model', default='/data/yutao/Project/weights/BGNet/x_admin.cluster.localRN-0.8BGNet_bs_14_epoch_9.pth', help='sr pretrained base model')
parser.add_argument('--save', default=False, action='store_true', help='If save test images')
parser.add_argument('--save_path', type=str, default='./test_results')
parser.add_argument('--input_size', type=int, default=256, help='input image size')
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=0.1)


opt = parser.parse_args()


def eval():
    model.eval()
    model.generator.eval()
    count = 1
    avg_psnr, avg_ssim, avg_l1 = 0., 0., 0.
    for batch in testing_data_loader:
        gt, mask, index = batch
        t_io2 = time.time()
        if cuda:
            gt = gt.cuda()
            mask = mask.cuda()


        ## The test or ensemble test
        # t0 = time.time()
        with torch.no_grad():
            prediction = model.generator(gt, mask)
        # t1 = time.time()
        # print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))

        # Save the video frames
        batch_avg_psnr, batch_avg_ssim, batch_avg_l1 = evaluate_batch(
            batch_size=opt.bs,
            gt_batch=gt,
            pred_batch=prediction,
            mask_batch=mask,
            save=opt.save,
            path=opt.save_path,
            count=count,
            index=index
            )

        # avg_psnr = (avg_psnr * (count - 1) + batch_avg_psnr) / count
        avg_psnr = avg_psnr + ((batch_avg_psnr- avg_psnr) / count)
        avg_ssim = avg_ssim + ((batch_avg_ssim- avg_ssim) / count)
        avg_l1 = avg_l1 + ((batch_avg_l1- avg_l1) / count)

        print(
            "Number: %05d" % (count * opt.bs),
            " | Average: PSNR: %.4f" % (avg_psnr),
            " SSIM: %.4f" % (avg_ssim),
            " L1: %.4f" % (avg_l1),
            "| Current batch:", count,
            " PSNR: %.4f" % (batch_avg_psnr),
            " SSIM: %.4f" % (batch_avg_ssim),
            " L1: %.4f" % (batch_avg_l1), flush=True
        )

        count+=1




def save_img(path, name, img):
    # img (H,W,C) or (H,W) np.uint8
    skimage.io.imsave(path+'/'+name+'.png', img)

def PSNR(pred, gt, shave_border=0):
    return compare_psnr(pred, gt, data_range=255)
    # imdff = pred - gt
    # rmse = math.sqrt(np.mean(imdff ** 2))
    # if rmse == 0:
    #     return 100
    # return 20 * math.log10(255.0 / rmse)

def L1(pred, gt):
    return np.mean(np.abs((np.mean(pred,2) - np.mean(gt,2))/255))

def SSIM(pred, gt, data_range=255, win_size=11, multichannel=True):
    return compare_ssim(pred, gt, data_range=data_range, \
    multichannel=multichannel, win_size=win_size)

def evaluate_batch(batch_size, gt_batch, pred_batch, mask_batch, save=False, path=None, count=None, index=None):
    if save:
        input_batch = gt_batch * (1 - mask_batch) + pred_batch * mask
        input_batch = (input_batch.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
        mask_batch = (mask_batch.detach().permute(0,2,3,1).cpu().numpy()[:,:,:,0]*255).astype(np.uint8)

    gt_batch = (gt_batch.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
    pred_batch = (pred_batch.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)

    psnr, ssim, l1 = 0., 0., 0.
    for i in range(batch_size):
        gt, pred, name = gt_batch[i], pred_batch[i], index[i]

        psnr += PSNR(pred, gt)
        ssim += SSIM(pred, gt)
        l1 += L1(pred, gt)

        if save:
            save_img(path, count+'_'+name+'_input', input_batch[i])
            save_img(path, count+'_'+name+'_mask', mask_batch[i])
            save_img(path, count+'_'+name+'_output', pred_batch[i])
            save_img(path, count+'_'+name+'_gt', gt_batch[i])

    return psnr/batch_size, ssim/batch_size, l1/batch_size



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


if __name__ == '__main__':
    ## Set the GPU mode
    gpus_list=range(opt.gpus)
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


    # Model
    model = InpaintingModel(g_lr=opt.lr, d_lr=(0.1 * opt.lr), l1_weight=opt.l1_weight, gan_weight=opt.gan_weight, iter=0, threshold=opt.threshold)
    print('---------- Networks architecture -------------')
    print("Generator:")
    print_network(model.generator)
    print("Discriminator:")
    print_network(model.discriminator)
    print('----------------------------------------------')

    if cuda:
        model = model.cuda()
        model.generator = torch.nn.DataParallel(model.generator, device_ids=gpus_list)
        model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=gpus_list)

    pretained_model = torch.load(opt.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretained_model)
    print('Pre-trained model is loaded.')

    # Datasets
    print('===> Loading datasets')
    testing_data_loader = build_dataloader(
        flist=opt.img_flist,
        mask_flist=opt.mask_flist,
        augment=False,
        training=False,
        input_size=opt.input_size,
        batch_size=opt.bs,
        num_workers=opt.threads,
        shuffle=True
    )
    print('===> Loaded datasets')

    ## Eval Start!!!!
    eval()
