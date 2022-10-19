import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import clip_gradient, AvgMeter
from keras import backend as K

from glob import glob
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F
import cv2
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
import math

import copy
import os.path as osp
import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__

from mmseg.models.segmentors import CaraSegUPer_ver2 as UNet





class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        # image = imread(img_path)
        # mask = imread(mask_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        # name = self.img_paths[idx].split('/')[-1]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352)) 

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)
    

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision +K.epsilon())


class FocalLossV1(nn.Module):
    
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


def train(train_loader, model, optimizer, epoch, lr_scheduler, deep=False):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        if epoch <= 1:
                optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * init_lr
        else:
            lr_scheduler.step()

        for rate in size_rates: 
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(trainsize_init*rate/32)*32)
            images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            map4, map3, map2, map1 = model(images)
            map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
            # ---- metrics ----
            dice_score = dice_m(map2, gts)
            iou_score = iou_m(map2, gts)
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, batchsize)
                dice.update(dice_score.data, batchsize)
                iou.update(iou_score.data, batchsize)

        # ---- train visualization ----
        if i == total_step:
            print('{} Training Epoch [{:03d}/{:03d}], '
                    '[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]'.
                    format(datetime.now(), epoch, num_epochs,\
                            loss_record.show(), dice.show(), iou.show()))

    ckpt_path = save_path + 'last.pth'
    print('[Saving Checkpoint:]', ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)

    log = OrderedDict([
        ('loss', loss_record.show()), ('dice', dice.show()), ('iou', iou.show()),
    ])

    return log


def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true)+np.sum(y_pred)-intersection
    return intersection/(union+K.epsilon())

def get_scores(gts, prs):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)        
    
    print(f"scores: dice={mean_dice}, miou={mean_iou}, precision={mean_precision}, recall={mean_recall}")

    return (mean_iou, mean_dice, mean_precision, mean_recall)



def inference(model):
    print("#"*20)
    model.eval()
    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    for dataset_name in dataset_names:
        data_path = f'/home/s/thuannh/dataset/scenario_4/all_datasets/TestDataset/{dataset_name}'
        print(data_path)
        X_test = glob('{}/images/*'.format(data_path))
        X_test.sort()
        y_test = glob('{}/masks/*'.format(data_path))
        y_test.sort()

        test_dataset = Dataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        print('Dataset_name:', dataset_name)
        tp_all = 0
        fp_all = 0
        fn_all = 0
        mean_iou = 0
        gts = []
        prs = []
        for i, pack in enumerate(test_loader, start=1):
            image, gt = pack
            # name = name[0]
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res, res2, res3, res4 = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pr = res.round()
            # cv2.imwrite(os.path.join(save_path, dataset_name, name), res)
            gts.append(gt)
            prs.append(pr)
        get_scores(gts, prs)
    print("#"*20)
    
    
init_lr = 1e-4
batchsize = 8
trainsize_init = 352
clip = 0.5
num_epochs= 20
train_save = 'PolypFormerB3_lan1'

save_path = 'snapshots/{}/'.format(train_save)
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
else:
    print("Save path existed")
#     sys.exit(1)


log = pd.DataFrame(index=[], columns=[
    'epoch', 'lr', 'loss', 'dice', 'iou', 'val_loss', 'val_dice', 'val_iou'
])
train_img_paths = []
train_mask_paths = []
train_img_paths = glob('/home/s/thuannh/dataset/scenario_4/all_datasets/TrainDataset/image/*')
train_mask_paths = glob('/home/s/thuannh/dataset/scenario_4/all_datasets/TrainDataset/mask/*')
train_img_paths.sort()
train_mask_paths.sort()

train_dataset = Dataset(train_img_paths, train_mask_paths)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

total_step = len(train_loader)






model = UNet(backbone=dict(
                type='mit_b3',
                style='pytorch'), 
            decode_head=dict(
                type='UPerHead',
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                channels=128,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                decoder_params=dict(embed_dim=768),
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
            neck=None,
            auxiliary_head=None,
            train_cfg=dict(),
            test_cfg=dict(mode='whole'),
            pretrained='pretrained/mit_b3.pth').cuda()

# ---- flops and params ----
params = model.parameters()
optimizer = torch.optim.Adam(params, init_lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                    T_max=len(train_loader)*num_epochs,
                                    eta_min=init_lr/1000)

start_epoch = 1
ckpt_path = ''
if ckpt_path != '':
    log = pd.read_csv(ckpt_path.replace('last.pth', 'log.csv'))
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])

print("#"*20, f"Start Training", "#"*20)
for epoch in range(start_epoch, num_epochs+1):
    train_log = train(train_loader, model, optimizer, epoch, lr_scheduler)

    log_tmp = pd.Series([epoch, optimizer.param_groups[0]["lr"], 
            train_log['loss'].item(), train_log['dice'].item(), train_log['iou'].item(),  
    ], index=['epoch', 'lr', 'loss', 'dice', 'iou'])
    log = log.append(log_tmp, ignore_index=True)
    log.to_csv(f'snapshots/{train_save}/log.csv', index=False)

    if epoch >= num_epochs-20:
        inference(model)
