from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import torchvision.transforms as transforms
#from PIL import Image
import cv2
import re
import pandas as pd
import os

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


class BasicDataset(Dataset):
	pass

class BasicDataset_split_csv(Dataset):
	def __init__(self, transform, if_split_csv, split_type, args):
		self.transform = transform
		self.if_split_csv = if_split_csv
		self.split_type = split_type
		self.args = args

		assert os.path.exists(if_split_csv)
		self.split_df = pd.read_csv(if_split_csv)

		self.sub_ids = list((self.split_df.loc[self.split_df['train_type'] == self.split_type])['image_name'])
		self.sub_mask_ids = list((self.split_df.loc[self.split_df['train_type'] == self.split_type])['mask_name'])

	def __len__(self):
		return len(self.sub_ids)

	@classmethod
	def preprocess(cls, img_nd):
		if len(img_nd.shape) == 2:
			img_nd = np.expand_dims(img_nd, axis=2)


		return img_nd

	def preprocess_mask(self, img_nd):


		if img_nd.max() > 1:
			if self.args.mask_type == "multi_seg":
				pass
			else:	
				img_nd = img_nd / 255

		if self.args.mask_type == "sigmoid01":
			
			img_nd = (np.array(img_nd) > 0).astype(int)

		return img_nd

	def __getitem__(self, i):
		idx = self.sub_ids[i]
		mask_idx = self.sub_mask_ids[i]

		assert os.path.exists(idx) 
		assert os.path.exists(mask_idx)

		img_file = glob(idx)
		mask_file = glob(mask_idx)
		assert os.path.basename(mask_file[0]) == os.path.basename(img_file[0]), \
		        f'{mask_file}, and {img_file} are not paired!'

		names = {"img_file": img_file, "mask_file": mask_file}

		
		assert len(mask_file) == 1, \
		    f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
		assert len(img_file) == 1, \
		    f'Either no image or multiple images found for the ID {idx}: {img_file}'

		if self.args.image_channel == 1:
			img = cv2.imread(img_file[0], cv2.IMREAD_GRAYSCALE)
		elif self.args.image_channel == 3:
			img = cv2.imread(img_file[0], cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		if self.args.mask_channel == 1:
			mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)
			
		img = img.astype(np.uint8)
		img = self.preprocess(img)
		mask = self.preprocess_mask(mask)
		

		img = self.transform(image=img, mask=mask)
		
		mask = {"image":img["mask"]}

		return {'image': img, 'mask': mask, 'names': names}



