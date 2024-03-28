
import os
os.environ["OMP_NUM_THREADS"] = "1" 

import torch
import re 
import random
import numpy as p
import argparse
from shutil import copyfile
import configparser

def main_process(args):

	if args.TYPE == "train":
		from before_train import before_training
		from train import training

		before_train_dict = before_training(args=args)
		training(before_train_dict, args=args)

	elif args.TYPE == "eval":
		from eval import eval
		eval(args=args)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('main process')

	## compulsory
	parser.add_argument(
		"--TYPE", 
		default = "train", 
		choices=["train", "test", "eval", "predict", "visualise", "integrate"]
		)

	## alter cfg
	parser.add_argument("--process_name", type=str)
	parser.add_argument("--description", type=str, default="test")
	parser.add_argument("--check_point_path", type=str, default="test")
	
	parser.add_argument("--ava_device", type=int)
	
	parser.add_argument("--if_split_csv", type=str)
	parser.add_argument("--best_model_path", type=str)
	parser.add_argument("--log_path", type=str)

	parser.add_argument("--label_type", type=str, default="seg", choices=["seg"])

	parser.add_argument("--SEED", type=int, default=152)
	parser.add_argument("--repo_level", type=int, default=2, choices=[0,1,2])

	parser.add_argument("--New_size", type=int)
	parser.add_argument("--point_num", type=int)
	parser.add_argument("--point_value", type=str)
	parser.add_argument("--image_channel", type=int, default=3, choices=[1,3])
	parser.add_argument("--mask_channel", type=int, default=1, choices=[1])
	parser.add_argument("--mask_type", type=str, default="sigmoid01", choices=["sigmoid01"])

	parser.add_argument("--transformer_train", type=str, default="train2_color")
	parser.add_argument("--transformer_val", type=str, default="basic_color")

	parser.add_argument("--batch_size", type=int)
	parser.add_argument("--num_workers", type=int, default=0)

	parser.add_argument("--model_name", type=str)
	parser.add_argument("--pretrained", type=bool, default=True, choices=[True, False])
	parser.add_argument("--snapshot", type=str, default=None)
	parser.add_argument("--freeze_type", type=str, default="_FZ0", choices=["_FZ0", "_FZ1", "_FZ2", "_FZ3", "_FZ4"])

	#parser.add_argument("--loss_type", type=str, default="BCELoss", choices=["BCELoss"])
	parser.add_argument("--loss_type", type=str, default="BCEDiceLoss", choices=["BCELoss", "CELoss", "BCEDiceLoss"])
	parser.add_argument("--optimizer_type", type=str, default="Adam")

	parser.add_argument("--check_num_per_epoch", type=int, default=1)
	parser.add_argument("--start_epoch", type=int, default=0)
	parser.add_argument("--TOTAL_EPOCH", type=int)
	parser.add_argument("--INITIAL_LR", type=float)
	parser.add_argument("--scheduler_type", type=str, default="CosineAnnealingLR_200_1e-7")

	args = parser.parse_args()
	
	args.best_model_path = args.best_model_path+args.process_name+"/"
	args.log_path = args.log_path+args.process_name+"/"
	args.New_size = [args.New_size, args.New_size]
	args.point_value = [int(i_) for i_ in args.point_value.split("-")]

	args.CALR_para = [int(args.scheduler_type.split("_")[1]), float(args.scheduler_type.split("_")[2])]
	args.scheduler_type = args.scheduler_type.split("_")[0]
	
	print(vars(args))
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ava_device)

	main_process(args)



























	