import pandas as pd
import os
import re 
import random
import numpy as np
import itertools
import cv2
import json
import shutil

import imageio
import re
from scipy.ndimage import distance_transform_edt

import matplotlib.pyplot as plt

def sorted_nicely( l ): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

def find_image_w0w1h0h1(img):
	mask = (img > 20) * 1.0
	mask[:20,:20,:] = 0
	mask = np.sum(mask, axis = 2)
	
	w_list = np.sum(mask, axis = 0)
	
	h_list = np.sum(mask, axis = 1)

	for i, w in enumerate(w_list):
		
		if w>10:
			w0 = i
			break
	for i, w in enumerate(w_list[::-1]):
		if w>10:
			w1 = len(w_list)-i
			break
	for i, h in enumerate(h_list):
		if h>10:
			h0 = i
			break
	for i, h in enumerate(h_list[::-1]):
		if h>10:
			h1 = len(h_list)-i
			break
	
	
	return w0,w1,h0,h1

def cut_and_pad_img(img,w0,w1,h0,h1,fill_pix):
	if len(img.shape) == 3:
		cut_img = img[h0:h1,w0:w1,:]
		
		max_wh = max(cut_img.shape)
		
		if fill_pix == 0:
			img_pad = np.zeros((max_wh, max_wh, 3))
		elif fill_pix == 255:
			img_pad = np.zeros((max_wh, max_wh, 3))
			img_pad = img_pad + 255
		img_pad[0:cut_img.shape[0], 0:cut_img.shape[1], :] = cut_img[:,:,:]
		
	elif len(img.shape) == 2:
		cut_img = img[h0:h1,w0:w1]
		
		max_wh = max(cut_img.shape)
		
		if fill_pix == 0:
			img_pad = np.zeros((max_wh, max_wh))
		elif fill_pix == 255:
			img_pad = np.zeros((max_wh, max_wh))
			img_pad = img_pad + 255
		img_pad[0:cut_img.shape[0], 0:cut_img.shape[1]] = cut_img[:,:]
		
	return img_pad.astype(np.uint8), max_wh

output_dict = {"image_name":[], "train_type":[], "mask_name":[]}
output_csv_path = "/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/split_Kvasir_1024.csv"





img_path = "/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/Kvasir/images/"
mask_path = "/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/Kvasir/masks/"
img_list = sorted_nicely([i for i in os.listdir(img_path) if not i.startswith(".")])
mask_list = sorted_nicely([i for i in os.listdir(mask_path) if not i.startswith(".")])

img_pre_path = "/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/Kvasir/images_1024/"
mask_pre_path = "/mnt/DATA-1/SAM_GROUP/Datasets/polyp/TestDataset/Kvasir/masks_1024/"
os.makedirs(img_pre_path, exist_ok = True)
os.makedirs(mask_pre_path, exist_ok = True)
print(len(img_list),len(mask_list))
for img, mask in zip(img_list, mask_list):

	loaded_img = cv2.imread(img_path + img, cv2.IMREAD_COLOR)
	loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
	#pad 
	w0,w1,h0,h1 = find_image_w0w1h0h1(loaded_img)
	loaded_img, _ = cut_and_pad_img(loaded_img, w0,w1,h0,h1, 0)
	#resize
	loaded_img = cv2.resize(loaded_img, (1024, 1024), interpolation = cv2.INTER_AREA)
	imageio.imwrite(img_pre_path + img, loaded_img)

	loaded_mask = cv2.imread(mask_path + mask, cv2.IMREAD_GRAYSCALE)
	loaded_mask, _ = cut_and_pad_img(loaded_mask, w0,w1,h0,h1, 0)
	#resize
	loaded_mask = cv2.resize(loaded_mask, (1024, 1024), interpolation = cv2.INTER_AREA)
	imageio.imwrite(mask_pre_path + img, ((loaded_mask>127)*255).astype(np.uint8))

img_pre_list = sorted_nicely([i for i in os.listdir(img_pre_path) if not i.startswith(".")])
mask_pre_list = sorted_nicely([i for i in os.listdir(mask_pre_path) if not i.startswith(".")])

for i,j in zip(img_pre_list, mask_pre_list):

	print(i,j)
	assert i == j

for i,j in zip(img_pre_list, mask_pre_list):
	output_dict["image_name"].append(img_pre_path+i)
	output_dict["mask_name"].append(mask_pre_path+j)
	output_dict["train_type"].append("val")

submit_df = pd.DataFrame.from_dict(output_dict)
submit_df.to_csv(output_csv_path, index=False)











