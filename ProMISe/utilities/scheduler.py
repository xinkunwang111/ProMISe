import torch.nn as nn
from torch.optim import lr_scheduler

def build_scheduler(scheduler_type, optimizer, args, for_who = "Generator"):
	if scheduler_type == "LR":
		scheduler = False
	elif scheduler_type == "CosineAnnealingLR":
		if for_who == "Generator":
			scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.CALR_para[0], args.CALR_para[1])
		elif for_who == "Discriminator":
			scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.CALR_para_D[0], args.CALR_para_D[1])
	elif scheduler_type == "CosineAnnealingWarmRestarts":
		scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.CAWR_para[0], T_mult=args.CAWR_para[1])
	
	return scheduler