import pandas as pd
import os
import re 
import random
#import torch
import numpy as np
import json
import math
import cv2

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_model(model_conv, load_dir):
    print("checkpoint load_dir: ", load_dir)
    checkpoint = torch.load(load_dir)
    model_conv.load_state_dict(checkpoint['state_dict'])
    print("checkpoint has been loaded")

    return model_conv

def get_chr_area(im_gray):
    im_gray = im_gray.astype('bool')
    img_area = np.sum(im_gray) 
    return img_area

from skimage import measure, metrics

def mse(imageA, imageB):
    
    # NOTE: the two images must have the same dimension
    imageA = (imageA>0)
    imageB = (imageB>0)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) 
    err /= float(imageA.shape[0] * imageA.shape[1]) 
    return err

def mae(imageA, imageB):
    # NOTE: the two images must have the same dimension
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float"))) 
    err /= float(imageA.shape[0] * imageA.shape[1]) 
    return err

def compare_images(imageA, imageB, title = None, showimage = False):
    
    imageA = (imageA>0)
    imageB = (imageB>0)
    m = mae(imageA, imageB)
    try:
        s = measure.compare_ssim(imageA, imageB, multichannel=False)
    except:
        s = metrics.structural_similarity(imageA, imageB, multichannel=False)
    
    if showimage == True:
        
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
        
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap = plt.cm.gray)
        plt.axis("off")
        
        
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap = plt.cm.gray)
        plt.axis("off")
        
        plt.show()
    else:
        pass
    
    return m, s


def compare_psnr(img1, img2):
    if np.max(img1) > 1.0:
        img1 = img1/255.0
    if np.max(img2) > 1.0:
        img2 = img2/255.0

    
    mse = mse(image1, image2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)
   
def compare_IOU(img1, img2):

    img1_mask = (img1>0)
    img2_mask = (img2>0)

    intersection = (img1_mask & img2_mask)*1.0
    union = (img1_mask | img2_mask)*1.0

    if union.sum() == 0:
        IOU = 0
    else:
        IOU = intersection.sum()/union.sum()

    if img2_mask.sum() == 0:
        Precision = 0
    else:
        Precision = intersection.sum()/img2_mask.sum() 

    if img1_mask.sum() == 0:
        Recall = 0
    else:
        Recall = intersection.sum()/img1_mask.sum() 

    return IOU, Precision, Recall

def compare_dice(img1, img2):

    img1_mask = (img1>0)
    img2_mask = (img2>0)

    intersection = (img1_mask & img2_mask)*1.0 

    Dice = (2*intersection.sum())/(img1_mask.sum()+img2_mask.sum())
    
    return Dice

def StructureMeasure(GT, prediction):
   
    if prediction.dtype != float:
       prediction = prediction.astype(float)

    if prediction.dtype != float:
       raise TypeError("prediction should be of type: float")

    if (np.max(prediction) > 1) or (np.min(prediction) < 0):
       raise ValueError("prediction should be in the range of [0, 1]")

    if not isinstance(GT, np.ndarray) or GT.dtype != bool:
       GT = GT.astype(bool)

    # Check input
    if not isinstance(prediction, np.ndarray) or prediction.dtype != np.float64:
        raise TypeError("The prediction should be a double type numpy array.")
    if prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("The prediction should be in the range of [0, 1].")
    if not isinstance(GT, np.ndarray) or GT.dtype != bool:
        raise TypeError("GT should be a logical type numpy array.")
    
    y = np.mean(GT)

    if y == 0:  # If the GT is completely black
        x = np.mean(prediction)
        Q = 1.0 - x  # Only calculate the area of intersection
    elif y == 1:  # If the GT is completely white
        x = np.mean(prediction)
        Q = x  # Only calculate the area of intersection
    else:
        alpha = 0.5
        Q = alpha * S_object(prediction, GT) + (1 - alpha) * S_region(prediction, GT)
        Q = max(Q, 0)  # Ensure the similarity score is not negative

    return Q


def S_object(prediction, GT):
    # Compute the similarity of the foreground at the object level
    prediction_fg = prediction.copy()
    prediction_fg[np.logical_not(GT)] = 0
    O_FG = Object(prediction_fg, GT)

    # Compute the similarity of the background
    prediction_bg = 1.0 - prediction
    prediction_bg[GT] = 0
    O_BG = Object(prediction_bg, np.logical_not(GT))

    # Combine the foreground measure and background measure together
    u = np.mean(GT)
    Q = u * O_FG + (1 - u) * O_BG

    return Q

def Object(prediction, GT):
    # Check the input
    if prediction.size == 0:
        score = 0
        return score
    if prediction.dtype == int:
        prediction = prediction.astype(float)
    if prediction.dtype != float:
        raise TypeError("prediction should be of type: float")
    if (np.max(prediction) > 1) or (np.min(prediction) < 0):
        raise ValueError("prediction should be in the range of [0, 1]")
    if not isinstance(GT, np.ndarray) or GT.dtype != bool:
        raise TypeError("GT should be of type: bool")

    # Compute the mean of the foreground or background in prediction
    x = np.mean(prediction[GT])

    # Compute the standard deviations of the foreground or background in prediction
    sigma_x = np.std(prediction[GT])

    score = 2.0 * x / (x**2 + 1.0 + sigma_x + np.finfo(float).eps)

    #print('object Score:',score)

    return score

def S_region(prediction, GT):

    X, Y = centroid(GT)
    
    # Divide GT into 4 regions
    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = divideGT(GT, X, Y)
    
    # Divide prediction into 4 regions
    prediction_1, prediction_2, prediction_3, prediction_4 = Divideprediction(prediction, X, Y)
    
    # Compute the ssim score for each region
    Q1 = ssim(prediction_1, GT_1)
    Q2 = ssim(prediction_2, GT_2)
    Q3 = ssim(prediction_3, GT_3)
    Q4 = ssim(prediction_4, GT_4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def centroid(GT):
    rows, cols = GT.shape

    if np.sum(GT) == 0:
        X = round(cols / 2)
        Y = round(rows / 2)
    else:
        dGT = GT.astype(float)

        ones_vector = np.ones((rows, 1))
        row_vector = np.arange(1, cols + 1)
        x = ones_vector * row_vector

        row_vector = np.arange(1, rows + 1)
        ones_vector = np.ones((1, cols))
        y = row_vector[:, np.newaxis] * ones_vector

        # Calculate the total area by summing all elements in dGT
        area = np.sum(dGT)

        # Calculate X and Y using element-wise multiplication and summation
        X = round(np.sum(np.sum(dGT * x)) / area)
        Y = round(np.sum(np.sum(dGT * y)) / area)

        # Convert X and Y to integers if needed
        X = int(X)
        Y = int(Y)

    return X, Y

def divideGT(GT, X, Y):
    hei, wid = GT.shape
    area = wid * hei

    LT = GT[:Y, :X]
    RT = GT[:Y, X:]
    LB = GT[Y:, :X]
    RB = GT[Y:, X:]

    w1 = (X * Y) / area
    w2 = ((wid - X) * Y) / area
    w3 = (X * (hei - Y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return LT, RT, LB, RB, w1, w2, w3, w4

def Divideprediction(prediction, X, Y):
    hei, wid = prediction.shape

    LT = prediction[:Y, :X]
    RT = prediction[:Y, X:]
    LB = prediction[Y:, :X]
    RB = prediction[Y:, X:]

    return LT, RT, LB, RB

def ssim(prediction,dGT):
    hei, wid = prediction.shape
    N = wid * hei

    x = np.mean(prediction)
    y = np.mean(dGT)

    sigma_x2 = np.sum((prediction - x) ** 2) / (N - 1 + np.finfo(float).eps)
    sigma_y2 = np.sum((dGT - y) ** 2) / (N - 1 + np.finfo(float).eps)
    sigma_xy = np.sum((prediction - x) * (dGT - y)) / (N - 1 + np.finfo(float).eps)

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(float).eps)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

def EnhancedAlignment(GT, FM):
    
    FM = np.array(FM, dtype=bool)
    
    GT = np.array(GT, dtype=bool)

    if not isinstance(GT, np.ndarray) or GT.dtype != bool:
       GT = GT.astype(bool)
    
    dFM = FM.astype(float)
    dGT = GT.astype(float)
    
    if np.sum(GT) == 0:
        enhanced_matrix = 1.0 - dFM  
    elif np.sum(~GT) == 0:
        enhanced_matrix = dFM  
    else:
        align_matrix = AlignmentTerm(dFM, dGT)
        enhanced_matrix = EnhancedAlignmentTerm(align_matrix)
    
    w, h = GT.shape
    score = np.sum(enhanced_matrix) / (w * h - 1 + np.finfo(float).eps)
    return score

def AlignmentTerm(dFM, dGT):
    
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)
    
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT
    
    align_Matrix = 2.0 * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + np.finfo(float).eps)
    return align_Matrix

def EnhancedAlignmentTerm(align_Matrix):
    
    enhanced = ((align_Matrix + 1) ** 2) / 4
    return enhanced

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter

def WFb(GT,FG):

    # Check input

    if not isinstance(FG, np.ndarray) or FG.dtype != np.float64:
       # raise TypeError("FG should be of type: float")
       FG = FG.astype(float)

    if not isinstance(GT, np.ndarray) or GT.dtype != bool:
       GT = GT.astype(bool)

    if FG.min() < 0 or FG.max() > 1:
        raise ValueError("FG should be in the range of [0, 1]")
    
    dGT = GT.astype(float)
    E = np.abs(FG - dGT) 

    from scipy.ndimage import morphology

    Dst = distance_transform_edt(~GT)
    flattened_GT = GT.ravel()
    flattened_Dst = Dst.ravel()

    IDXT = np.zeros_like(dGT)
    mindst=200000 
    nonzero_indices_row, nonzero_indices_col= np.nonzero(Dst)
    minx=0
    miny=0

    for i in range(GT.shape[0]):
        for j in range (GT.shape[1]):
            if GT[i,j] != 0:
                IDXT[i,j]= i*(GT.shape[1]) +j
            else:
               for x, y in zip(nonzero_indices_row, nonzero_indices_col):
                  newdst=(x-i)**2+(y-j)**2
                  if newdst< mindst:
                      mindst=newdst
                      minx=x
                      miny=y
               IDXT[i,j]= minx*(GT.shape[1]) +miny
    
    K = gaussian_filter(np.zeros((7, 7)), sigma=5)
    Et = E.copy()
    IDXT= IDXT.astype(int)
    
    row_indices, col_indices = np.nonzero(~GT) 
    linear_indices = IDXT[row_indices, col_indices] 
    linear_indices.astype(int)
    
    idx = np.unravel_index(linear_indices, GT.shape) 
    Et[row_indices, col_indices] = Et[idx]

    from scipy.ndimage import convolve 
    EA = convolve(Et, K) 

    MIN_E_EA = E.copy()
    MIN_E_EA[GT & (EA < E)] = EA[GT & (EA < E)] 

    B = np.ones_like(GT)
    B[np.logical_not(GT)] = 2.0 - np.exp(np.log(1 - 0.5) / 5 * Dst[np.logical_not(GT)]) 

    Ew = MIN_E_EA * B 

    TPw = np.sum(dGT) - np.sum(Ew[GT]) 
    FPw = np.sum(Ew[np.logical_not(GT)])
    
    R = 1 - np.mean(Ew[GT])  
    P = TPw / (TPw + FPw + np.finfo(float).eps)  

    if R + P !=0:
        Q = (2) * (R * P) / (R + P + np.finfo(float).eps)  # Beta=1
    else:
        Q=0
    
    return Q







