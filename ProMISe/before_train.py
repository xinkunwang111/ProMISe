#import cfg

from utilities.data_loader import BasicDataset, BasicDataset_split_csv
from utilities.build_network import build_network
from utilities.optimizer import build_optimizer
from utilities.transformers import *

import pandas as pd
import os
import re 
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

def init_func(m): 
	if isinstance(m, nn.Conv2d):
		
		nn.init.kaiming_uniform_(m.weight)
		try:
			m.bias.data.fill_(0.01)
		except:
			pass
	if isinstance(m, nn.Linear): 
		
		nn.init.kaiming_uniform_(m.weight)
		try:
			m.bias.data.fill_(0.01)
		except:
			pass

def before_training(args):

	random.seed(args.SEED)
	np.random.seed(args.SEED)
	torch.manual_seed(args.SEED)
	torch.cuda.manual_seed(args.SEED)
	torch.cuda.manual_seed_all(args.SEED)
	#torch.backends.cudnn.deterministic = True
	#torch.backends.cudnn.benchmark = False
	print("fixed random seed:", args.SEED)
	print("init LR:", args.INITIAL_LR)
	print("batch_size:", args.batch_size)

	train_transforms = applied_transforms(args.New_size, applied_types = args.transformer_train)
	val_transforms = applied_transforms(args.New_size, applied_types = args.transformer_val)

	assert(os.path.exists(args.if_split_csv))
	print("split dataset by an existing csv:", args.if_split_csv)

	train_dataset = BasicDataset_split_csv(
							transform = train_transforms, 
							if_split_csv = args.if_split_csv,
							split_type = 'train',
							args = args)
	
	val_dataset = BasicDataset_split_csv(
							transform = val_transforms, 
							if_split_csv = args.if_split_csv,
							split_type = 'val',
							args = args)
	
	test_dataset = BasicDataset_split_csv(
							transform = val_transforms, 
							if_split_csv = args.if_split_csv,
							split_type = 'test',
							args = args)
	
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	
	
	print("-------------------------------------------------")
	print(len(train_dataset), len(val_dataset), len(test_dataset))
	print(len(train_loader), len(val_loader), len(test_loader))

	print("------------------------train loader-------------------------")
	i=0
	for tr_batch_i, batch in enumerate(train_loader):
		print(tr_batch_i, batch["names"])
		print(batch["image"]['image'].shape, batch["mask"]['image'].shape)
		i+=1
		if i >= 3:
			break
	print("-------------------------val loader------------------------")
	i=0
	for tr_batch_i, batch in enumerate(val_loader):
		print(tr_batch_i, batch["names"])
		print(batch["image"]['image'].shape, batch["mask"]['image'].shape)
		i+=1
		if i >= 3:
			break
	print("-------------------------test loader------------------------")
	i=0
	for tr_batch_i, batch in enumerate(test_loader):
		print(tr_batch_i, batch["names"])
		print(batch["image"]['image'].shape, batch["mask"]['image'].shape)
		i+=1
		if i >= 3:
			break


	model_conv = build_network(model_name = args.model_name, 
								  pretrained=args.pretrained,
								  snapshot = args.snapshot,
								  args = args)
	#model_conv.apply(init_func)
	print('initialize network')
	before_train_dict = {"model_conv":model_conv, "train_loader":train_loader, "val_loader":val_loader}

	return before_train_dict













