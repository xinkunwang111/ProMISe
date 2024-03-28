import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock

from timm.models.vision_transformer import PatchEmbed


def concat(output_feature, concat_type):
    if str(concat_type)=='last_layer':
        x=output_feature[3]

    elif str(concat_type)=='9_12':
        x=torch.cat((output_feature[3], output_feature[2]), dim=1)
        
    elif str(concat_type)=='6_9_12':
        x=torch.cat((output_feature[3], output_feature[2],output_feature[1]), dim=1)
        
    elif str(concat_type) =='all':
        x=torch.cat((output_feature[3], output_feature[2],output_feature[1], output_feature[0]) ,dim=1)
        
    return x

#-------------------------------------------------------------------------------------------------------------------
class FeatureMap(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans: int = 256,
        args=None):
        
        super().__init__()
      

        self.fusion_conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3,padding=1)
        self.fusion_conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3,padding=1)
        self.fusion_conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=3,padding=1)
        self.relu = nn.ReLU()

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x=self.fusion_conv3(self.relu(self.fusion_conv2(self.relu(self.fusion_conv1(x)))))
        
        return x

def FeatureMap12(in_chans=768,args = None):
    return FeatureMap(in_chans,args=args)

def FeatureMap12_9(in_chans=1536,args = None):
    return FeatureMap(in_chans,args=args)

def FeatureMap12_9_6(in_chans=2304,args = None):
    return FeatureMap(in_chans,args=args)

def FeatureMap12_9_6_3(in_chans=3072,args = None):
    return FeatureMap(in_chans,args=args)

#----------------------------------------------------------------------------------------------------
class Scale(nn.Module):
    def __init__(
        self,
        feature,
        dim = 256,
        args=None):
        
        super(Scale, self).__init__()

        self.feature = feature

        self.deconv_768_512 = nn.ConvTranspose2d(dim*3, dim*2, kernel_size=2, padding =2)
        self.deconv_768_512 = self.deconv_768_512.to('cuda')

        self.deconv_512_256 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, padding =2)
        self.deconv_512_256 = self.deconv_512_256.to('cuda')

        self.conv_1024_768 = nn.Conv2d(dim*4, dim*3, kernel_size=3,padding =1)
        self.conv_1024_768 = self.conv_1024_768.to('cuda')

        self.conv_768_512 = nn.Conv2d(dim*3, dim*2, kernel_size=3,padding =1)
        self.conv_768_512 = self.conv_768_512.to('cuda')

        self.conv_512_256 = nn.Conv2d(dim*2, dim, kernel_size=3,padding =1)
        self.conv_512_256 = self.conv_512_256.to('cuda')

        self.conv_512 = nn.Conv2d(dim*2, dim*2, kernel_size=3,padding =1)
        self.conv_512 = self.conv_512.to('cuda')

        self.conv_256 = nn.Conv2d(dim, dim, kernel_size=3,padding =1)
        self.conv_256 = self.conv_256.to('cuda')

        self.batchnorm_768 = nn.BatchNorm2d(dim*3)
        self.batchnorm_768 = self.batchnorm_768.to('cuda')

        self.batchnorm_512 = nn.BatchNorm2d(dim*2)
        self.batchnorm_512 = self.batchnorm_512.to('cuda')

        self.batchnorm_256 = nn.BatchNorm2d(dim)
        self.batchnorm_256 = self.batchnorm_256.to('cuda')

        self.relu = nn.ReLU()
        self.relu = self.relu.to('cuda')
   
   
    def forward(self, x: torch.Tensor, feature) -> torch.Tensor:
        if self.feature == '9':
            x = self.relu(self.batchnorm_512(self.conv_512(self.deconv_768_512(x))))
        elif self.feature == '6':
            x = self.relu(self.batchnorm_512(self.conv_512(self.deconv_768_512(x))))
            x = self.relu(self.batchnorm_256(self.conv_256(self.deconv_512_256(x))))
        elif self.feature == '12':
            x= self.deconv_768_512(x)
        elif self.feature == 'middle':
            x= self.deconv_512_256(x)
        elif self.feature == 'concat1':
            x =self.relu(self.batchnorm_768(self.conv_1024_768(x)))
            x =self.relu(self.batchnorm_512(self.conv_768_512(x)))
        elif self.feature == 'concat2':
            x =self.relu(self.batchnorm_256(self.conv_512_256(x)))
            x =self.relu(self.batchnorm_256(self.conv_256(x)))
        return x
            
        
class FeatureScale(nn.Module):
    def __init__(
        self,
        args=None):
        
        super(FeatureScale, self).__init__()

        self.feature6 = Scale(feature = '6')
        self.feature9 = Scale(feature = '9')
        self.feature12 = Scale(feature = '12')
        self.feature_concat1 = Scale(feature = 'concat1')
        self.feature_concat2 = Scale(feature = 'concat2')
        self.feature_middle = Scale(feature = 'middle')
   
   
    def forward(self, featuremap):
        key12 = featuremap[3]
        
        key9 = featuremap[2]
        key6 = featuremap[1]

        key12_scale = self.feature12(key12, feature = '12')
        key9_scale = self.feature9(key9, feature = '9')
        key6_scale = self.feature6(key6, feature = '6')
        concat12_9 = torch.cat((key12_scale, key9_scale), dim=1)
        conv12_9 = self.feature_middle(self.feature_concat1(concat12_9,feature = 'concat1'),feature = 'middle')
        concat12_9_6 = torch.cat((conv12_9, key6_scale), dim=1)
        conv12_9_6 = self.feature_concat2(concat12_9_6, feature = 'concat2')

        return conv12_9_6

