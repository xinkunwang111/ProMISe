
from utilities.data_loader import BasicDataset_split_csv
from utilities.build_network import build_network
from utilities.optimizer import build_optimizer
from utilities.scheduler import build_scheduler
from utilities.transformers import *
from utilities.general_utilities import *
from train import *

import argparse
import imageio

import pandas as pd
import os
import re
import random
import numpy as np
import cv2
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

def eval(args):
	random.seed(args.SEED)
	np.random.seed(args.SEED)
	torch.manual_seed(args.SEED)
	torch.cuda.manual_seed(args.SEED)
	torch.cuda.manual_seed_all(args.SEED)
	
	print("fixed random seed:", args.SEED)
	print("init LR:", args.INITIAL_LR)
	print("batch_size:", args.batch_size)

	train_transforms = applied_transforms(args.New_size, applied_types=args.transformer_train)
	val_transforms = applied_transforms(args.New_size, applied_types=args.transformer_val)

	assert (os.path.exists(args.if_split_csv))
	print("split dataset by an existing csv:", args.if_split_csv)

	train_dataset = BasicDataset_split_csv(
		transform=train_transforms,
		if_split_csv=args.if_split_csv,
		split_type='train',
		args=args)

	val_dataset = BasicDataset_split_csv(
		transform=val_transforms,
		if_split_csv=args.if_split_csv,
		split_type='val',
		args=args)

	test_dataset = BasicDataset_split_csv(
		transform=val_transforms,
		if_split_csv=args.if_split_csv,
		split_type='test',
		args=args)

	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	model_conv = build_network(model_name=args.model_name,
							   pretrained=args.pretrained,
							   snapshot=args.snapshot,
							   args = args )
	
	print('initialize network')
	before_train_dict = {"model_conv": model_conv, "train_loader": train_loader, "val_loader": val_loader}

	model_conv, _, val_loader = before_train_dict["model_conv"], before_train_dict["train_loader"], before_train_dict["val_loader"]
	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	checkpoint = torch.load(args.check_point_path)

	#checkpoint_model = checkpoint['state_dict']
	checkpoint_model = checkpoint
	# print("checkpoint_model.keys()", checkpoint_model.keys())
	for k in list(checkpoint_model.keys()):
		# retain only encoder up to before the embedding layer
		if k.startswith('ffn.lin'):
			checkpoint_model["ips_"+k] = checkpoint_model[k]
			# delete renamed or unused k
			del checkpoint_model[k]
	
	model_conv.load_state_dict(checkpoint,strict = False)  # ['state_dict'])
	model_conv.to(device, dtype=torch.float32)
	print(f"=> successfully loaded checkpoint ")

	cudnn.benchmark = True
	model_conv.eval()

	val_MAE, val_Dice, val_IOU,  = [], [],  []
	
	for val_batch_i, batch in enumerate(val_loader):
		imgs = batch['image']['image']
		true_masks = batch['mask']['image']
		names = batch['names']['mask_file'][0][0]
		imgs, true_masks = imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.float32)

		if args.loss_type == "BCEDiceLoss":  
			labels_torch = torch.as_tensor(np.array(args.point_value), dtype=torch.int, device=device)
			batch_dict = {"image": imgs, "labels_torch": labels_torch,"true_masks":true_masks}
			outputs_pred,ufo = model_conv(batch_dict, multimask_output=True)
			masks_pred = torch.cat([jj["masks"] for jj in outputs_pred], dim=0)
			masks_pred = masks_pred[:, 0:1, :, :]

		true_masks = F.interpolate(true_masks, (256, 256), mode="bilinear", align_corners=False)  

		if args.mask_type == 'sigmoid01':
			masks_pred = (masks_pred >= 0.5) * 1.0
			true_masks, masks_pred = true_masks.detach().cpu().numpy(), masks_pred.detach().cpu().numpy()  # 张量转换为 NumPy 数组（arrays）
			true_masks = np.reshape(true_masks, (true_masks.shape[2:]))
			masks_pred = np.reshape(masks_pred, (masks_pred.shape[2:]))
			kernel=np.ones((5,5),dtype=np.uint8)
			# masks_pred = cv2.morphologyEx(masks_pred, cv2.MORPH_CLOSE, kernel)
			masks_pred = cv2.morphologyEx(masks_pred, cv2.MORPH_OPEN, kernel)

		assert true_masks.shape == masks_pred.shape  
		m, s = compare_images(true_masks, masks_pred, title=None, showimage=False)
		val_MAE.append(m)  

		if args.mask_type == 'sigmoid01':
			masks_pred = (masks_pred * 255).astype(np.uint8)
			true_masks = (true_masks * 255).astype(np.uint8)

		IOU, _, _ = compare_IOU(true_masks, masks_pred)
		val_IOU.append(IOU) 

		Dice = compare_dice(true_masks, masks_pred)
		val_Dice.append(Dice)  

		val_avg_MAE = np.mean(val_MAE)

		val_avg_Dice = np.mean(val_Dice)
		val_avg_IOU = np.mean(val_IOU)

	print("MAE: %.3f, Dice: %.3f, IOU: %.3f, " % (val_avg_MAE, val_avg_Dice, val_avg_IOU))









