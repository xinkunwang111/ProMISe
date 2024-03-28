import albumentations
import albumentations.augmentations.transforms as transforms
## pip install albumentations==0.4.6 导入否则报错
#from albumentations.pytorch import ToTensor 
from albumentations.pytorch import ToTensor
import torchvision
import cv2

def applied_transforms(New_size, applied_types = None):
	## albumentations
	## can be used as basic traning and test set, without any augmentation
	if applied_types == None:
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    ToTensor()
		    ])

	
	elif applied_types == "basic_gray":
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    albumentations.Normalize(mean=[0.5],std=[0.5], max_pixel_value=255.0),
		    ToTensor()
		    ])

	elif applied_types == "basic_mask":
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    ToTensor()
		    ])

	elif applied_types == "basic_color":
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    albumentations.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
		    ToTensor()
		    ])

	elif applied_types == "basic_color_nonorm":
		data_transforms = albumentations.Compose([
		    albumentations.Resize(New_size[0], New_size[1]),
		    #albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
		    ToTensor()
		    ])

	elif applied_types == "train1_color":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			albumentations.RandomResizedCrop(height = New_size[0], width = New_size[1], scale=(0.95, 1.05), ratio=(0.95, 1.05), p=0.25),
			albumentations.ShiftScaleRotate(shift_limit=0.0625,
										scale_limit=0.05,
										rotate_limit=10,
										p=0.25),
			albumentations.OneOf([
				albumentations.Blur(blur_limit=5),
				albumentations.GaussianBlur(blur_limit=5),
				albumentations.MedianBlur(blur_limit=5),
				albumentations.MotionBlur(blur_limit=5)
				], p=0.25),
			transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
		    albumentations.OneOf([
		        #albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(),
		        #albumentations.RandomBrightness(), albumentations.RandomContrast(),
		        albumentations.JpegCompression(), albumentations.GaussNoise()], p=0.25), 
			albumentations.CoarseDropout(p=0.1),
			albumentations.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
			ToTensor()
			])

	elif applied_types == "train2_color":
		data_transforms = albumentations.Compose([
			albumentations.Resize(New_size[0], New_size[1]),
			albumentations.RandomResizedCrop(height = New_size[0], width = New_size[1], scale=(0.95, 1.05), ratio=(0.95, 1.05), p=0.25),
			
			albumentations.HorizontalFlip(p=0.25),
			albumentations.VerticalFlip(p=0.25),
			#albumentations.IAAPerspective(scale=(0.05, 0.1), p=0.5),
			albumentations.ShiftScaleRotate(shift_limit=0.0625,
										scale_limit=0.05,
										rotate_limit=45,
										p=0.25),
			
			albumentations.OneOf([
				albumentations.Blur(blur_limit=5),
				albumentations.GaussianBlur(blur_limit=5),
				albumentations.MedianBlur(blur_limit=5),
				albumentations.MotionBlur(blur_limit=5)
				], p=0.25),
			transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
		    albumentations.OneOf([
		        albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(),
		        albumentations.JpegCompression(), albumentations.GaussNoise()], p=0.25), 
			
			#albumentations.Cutout(p=0.5),
			albumentations.CoarseDropout(p=0.1),
			albumentations.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
			ToTensor()
			])

	return data_transforms

