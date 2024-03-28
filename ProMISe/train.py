from utilities.data_loader import BasicDataset
from utilities.build_network import build_network
from utilities.optimizer import build_optimizer
from utilities.scheduler import build_scheduler
from utilities.transformers import *
from utilities.general_utilities import *

import pandas as pd
import os
import re
import random
import numpy as np
import cv2
import cv2
import shutil
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

def dice_loss(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = 2. * (intersection + 1) / (preds.sum(1) + trues.sum(1) + 1)

    if is_average:
        score = scores.sum() / num
        return torch.clamp(score, 0., 1.)
    else:
        return scores

class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        return 1 - dice_loss(F.sigmoid(input), target, weight=weight, is_average=self.size_average)

class BCEDiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss(size_average=size_average)

    def forward(self, input, target, weight=None):
        return nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average, weight=weight)(input, target) + self.dice(input, target, weight=weight)


def training(before_train_dict, args):  # , test_loader):
    model_conv, train_loader, val_loader = before_train_dict["model_conv"], before_train_dict["train_loader"], before_train_dict["val_loader"]

    os.makedirs(args.log_path, exist_ok=True)
    log_loss_file_name = args.log_path + "log_loss_" + args.process_name + '.csv'
   
    if os.path.exists(log_loss_file_name):
        os.makedirs(args.log_path + "/last_bak/", exist_ok=True)
        shutil.copy(log_loss_file_name, args.log_path + "/last_bak/" + "log_loss_" + args.process_name + '.csv')

    with open(log_loss_file_name, 'a') as log_file:
        log_file.seek(0)
        log_file.truncate()

    with open(log_loss_file_name, 'a') as log_file:
        log_file.write(
			'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' %
			("Epoch", "Batches", "train_avg_loss_"+args.loss_type, "val_avg_loss_"+args.loss_type, "MAE", "Dice", "IOU", "Sα", "Eφmax" ,"F_beta"   
			 ))
    
    check_Point = int(len(train_loader) / args.check_num_per_epoch)
    INITIAL_LR = args.INITIAL_LR
    Stop = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_conv.to(device, dtype=torch.float32)
    ## Freeze all SAM parameters
    if args.freeze_type == "_FZ0":
        for name, param in model_conv.named_parameters():
            if ("image_encoder" in name) or ("mask_decoder" in name) or ("prompt_encoder" in name):
                param.requires_grad = False
    ## Freeze prompt encoder
    elif args.freeze_type == "_FZ1":
        for name, param in model_conv.named_parameters():
            if ("image_encoder" in name) or ("mask_decoder" in name):
                param.requires_grad = False
    ## Freeze mask decoder
    elif args.freeze_type == "_FZ2":
        for name, param in model_conv.named_parameters():
            if ("image_encoder" in name) or ("prompt_encoder" in name):
                param.requires_grad = False
    ## Freeze prompt encoder and mask decoder
    elif args.freeze_type == "_FZ3":
        for name, param in model_conv.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = False
    ## Finetune all parameters
    elif args.freeze_type == "_FZ4":
        pass

    parameters = list(filter(lambda p: p.requires_grad, model_conv.parameters()))
    print("len(parameters)", len(parameters))
    print("model_conv", model_conv)
    
    criterion = BCEDiceLoss()
    optimizer = build_optimizer(args.optimizer_type, model_conv, INITIAL_LR)

    scheduler = build_scheduler(args.scheduler_type, optimizer, args)
    print(args.scheduler_type, "loaded")
    best_val_iou = 0

    print("start_epoch", args.start_epoch)

    for i in range(args.start_epoch, args.TOTAL_EPOCH):
        if Stop:
            print("Early stopping")
            break
        print("------------------------------------------------------")
        print("Training starts: Epoch ", i)
        print("------------------------------------------------------")

        train_loss = []
        check_num = -1
        for tr_batch_i, batch in enumerate(train_loader):
            model_conv.train()
            names = batch['names']['mask_file']
            imgs = batch['image']['image']
            true_masks = batch['mask']['image']
            imgs, true_masks = imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.float32)
            optimizer.zero_grad()

            labels_torch = torch.as_tensor(np.array(args.point_value), dtype=torch.int, device=device)
            batch_dict = {"image": imgs, "labels_torch": labels_torch,"true_masks":true_masks}
            outputs_pred,middle= model_conv(batch_dict, multimask_output=True)
            masks_pred = torch.cat([jj["masks"] for jj in outputs_pred], dim=0)
            masks_pred = masks_pred[:, 0:1, :, :]

            true_masks = F.interpolate(true_masks, (256, 256), mode="bilinear", align_corners=False)
            loss=criterion(masks_pred, true_masks)
            assert criterion(masks_pred, true_masks).requires_grad == True

            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().cpu().item())

            if (tr_batch_i + 1) % check_Point == 0:
                check_num += 1
                model_conv.eval()

                val_loss, val_MAE, val_Dice, val_IOU, val_Sα, val_Eφmax= [], [], [], [], [], []
                val_Pre, val_Rec, val_F_beta= [], [], []
				
                for val_batch_i, batch in enumerate(val_loader):
                    imgs = batch['image']['image']
                    true_masks = batch['mask']['image']
                    names = batch['names']['mask_file'][0][0]
                    imgs, true_masks = imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.float32)
                    
                    labels_torch = torch.as_tensor(np.array(args.point_value), dtype=torch.int, device=device)
                    batch_dict = {"image": imgs, "labels_torch": labels_torch,"true_masks":true_masks}
                    outputs_pred,middle = model_conv(batch_dict, multimask_output=True)
                    masks_pred = torch.cat([jj["masks"] for jj in outputs_pred], dim=0)
                    masks_pred = masks_pred[:, 0:1, :, :]

                    true_masks = F.interpolate(true_masks, (256, 256), mode="bilinear", align_corners=False)

                    loss = criterion(masks_pred, true_masks)
                    val_loss.append(loss.detach().cpu().item())

                    if args.mask_type == 'sigmoid01':
                        masks_pred = (masks_pred >= 0.5) * 1.0
                        true_masks, masks_pred = true_masks.detach().cpu().numpy(), masks_pred.detach().cpu().numpy()
                        true_masks = np.reshape(true_masks, (true_masks.shape[2:]))
                        masks_pred = np.reshape(masks_pred, (masks_pred.shape[2:]))
                        kernel = np.ones((5, 5), dtype=np.uint8)
                        # masks_pred = cv2.morphologyEx(masks_pred, cv2.MORPH_CLOSE, kernel)
                        masks_pred = cv2.morphologyEx(masks_pred, cv2.MORPH_OPEN, kernel)

                    assert true_masks.shape == masks_pred.shape 
                    m,s = compare_images(true_masks, masks_pred, title = None, showimage = False)
                    val_MAE.append(m) 

                    S= StructureMeasure(true_masks, masks_pred)
                    val_Sα.append(S) 
                    E= EnhancedAlignment(true_masks, masks_pred)
                    val_Eφmax.append(E) 
                    
                    if args.mask_type == 'sigmoid01':
                        masks_pred = (masks_pred*255).astype(np.uint8)
                        true_masks = (true_masks*255).astype(np.uint8)

                    IOU, Precision, Recall = compare_IOU(true_masks, masks_pred)
                    val_IOU.append(IOU) 
                    val_Pre.append(Precision)
                    val_Rec.append(Recall)

                    if 0.3*Precision+Recall!=0:
                        val_F_beta.append(1.3*(Precision*Recall)/(0.3*Precision+Recall+ np.finfo(float).eps)) #这里加F_beta-------------------------------问一下
                    else:
                        val_F_beta.append(0)

                    Dice = compare_dice(true_masks, masks_pred)
                    val_Dice.append(Dice)

                train_avg_loss = np.mean(train_loss)
                val_avg_loss = np.mean(val_loss)
                val_avg_MAE = np.mean(val_MAE)

                val_avg_Dice = np.mean(val_Dice)
                val_avg_IOU = np.mean(val_IOU)

                val_avg_F_beta = np.mean(val_F_beta)

                val_avg_Sα = np.mean(val_Sα)
                val_avg_Eφmax = np.mean(val_Eφmax)

                val_avg_Pre = np.mean(val_Pre)
                val_avg_Rec = np.mean(val_Rec)

                print("Epoch %d, batches %d : train loss: %.3f, valid loss: %.3f, MAE: %.3f, Dice: %.3f, IOU: %.3f, Sα: %.3f, Eφmax: %.3f, F_beta: %.3f"%(i, tr_batch_i+1, train_avg_loss, val_avg_loss, val_avg_MAE, val_avg_Dice, val_avg_IOU,  val_avg_Sα, val_avg_Eφmax, val_avg_F_beta ))
				
                with open(log_loss_file_name, 'a') as log_file:
                    log_file.write(
						'%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f\n' %
						(i, (tr_batch_i+1), train_avg_loss, val_avg_loss, val_avg_MAE, val_avg_Dice, val_avg_IOU, val_avg_Sα, val_avg_Eφmax ,val_avg_F_beta ))
				
                #------------------------------------------------------------------------------------------------------------------------------------------------

                if args.best_model_path != None:
                    os.makedirs(args.best_model_path, exist_ok=True)
                    torch.save(model_conv.state_dict(), args.best_model_path + 'model_val_last.pt')

                if val_avg_IOU > best_val_iou:
                    best_val_iou = val_avg_IOU
                    if args.best_model_path != None:
                        os.makedirs(args.best_model_path, exist_ok=True)
                        torch.save(model_conv.state_dict(), args.best_model_path + 'model_val_best_%d_%d_%.3f.pt' % (i, check_num, val_avg_IOU))

                    print(" --------- New best validation iou --------- ", best_val_iou)

                if scheduler:
                    print("Current_LR:", optimizer.param_groups[0]['lr'], scheduler.get_last_lr()[0])
                else:
                    print("Current_LR:", INITIAL_LR)

    if scheduler:
        scheduler.step()
