
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Optional, Type

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import numpy as np
import cv2
import os
from .image_encoder import *
from .position_embedding import PositionEmbeddingSine

class preprecessing(nn.Module):
    def __init__(self,dim):
        super(preprecessing,self).__init__()
        self.dim=dim
        self.conv1=nn.Conv2d(dim,dim*4,3,padding=1)
        self.conv2=nn.Conv2d(dim*4,dim,3,1,padding=1)
        self.conv3=nn.Conv2d(dim,dim,3,padding=1)
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
        self.relu3 = nn.ReLU()
    def forward(self,x):
        x=self.relu1(self.conv1(x))
        x=self.relu2(self.conv2(x))
        x=self.conv3(x)
        return x

class MaskformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate=0):
        super(MaskformerDecoderLayer, self).__init__()

        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate,batch_first=True)
        self.ffn = MLPBlock(hidden_dim, mlp_dim=4*hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, query_embedding,src,position_embedd,self_mask=None, cross_mask=None):
        cross_attention_output, _ = self.cross_attention(x+query_embedding,src+position_embedd, src, attn_mask=cross_mask)
        x = self.norm1(x + cross_attention_output)
        x=self.norm3(x+self.ffn(x))
        return x

class multiheadattention(nn.Module):
    def __init__(self,hidden_dim,num_heads,dropout_rate):
        super(multiheadattention,self).__init__()
        self.attn=nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate,batch_first=True)
        self.norm1=nn.LayerNorm(hidden_dim)

    def forward(self,x,key, value, self_mask=None, cross_mask=None):
        output,_=self.attn(x,key, value)
        x=self.norm1(x+output)
        return x

class deconv(nn.Module):
    def __init__(self,hidden_dim):
        super(deconv,self).__init__()
        self.deconv= nn.ConvTranspose2d(hidden_dim,hidden_dim,2,2)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1)
        self.batchnorm = nn.BatchNorm2d(hidden_dim)
        self.Relu = nn.ReLU()
        
    def forward(self,feature_map):
        feature_map = feature_map.permute(0,3,1,2)
        feature_map = self.deconv(feature_map)
        feature_map = self.conv(feature_map)
        feature_map = self.batchnorm(feature_map)
        feature_map = self.Relu(feature_map)

        feature_map = feature_map.permute(0,2,3,1)
        return feature_map

class maskformerdecoder_forward(nn.Module):
    def __init__(self,hidden_dim):
        super(maskformerdecoder_forward,self).__init__()
        self.sinsignal=PositionEmbeddingSine(hidden_dim//2,normalize=True)
    def forward(self,key,level_embedding):
        value=key
        
        key_po = key.permute(0, 3, 1, 2)
        
        value_po = value.permute(0, 3, 1, 2)
        key_po = self.sinsignal(key_po).flatten(2)
        value_po = self.sinsignal(value_po).flatten(2)

        key_po = key_po.permute(0, 2, 1)
        value_po = value_po.permute(0, 2, 1)
        
        value = value.view(value.shape[0], -1, value.shape[3])
        key = key.view(key.shape[0], -1, key.shape[3])
        
        src=key+level_embedding
        
        return src,key_po


class MaskformerDecoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, num_queries, dropout_rate=0.1):
        super(MaskformerDecoder, self).__init__()
        self.num_queried = num_queries

        self.query_embedding = nn.Embedding(num_queries,hidden_dim)
        self.query_feat=nn.Embedding(num_queries,hidden_dim)
        self.level_embedding = nn.Embedding(3, 768)

        self.layers = nn.ModuleList([
            MaskformerDecoderLayer(hidden_dim, num_heads) for layer in range(num_layers*3)
            ])

        self.process12=maskformerdecoder_forward(hidden_dim)
        self.process9=maskformerdecoder_forward(hidden_dim)
        self.process6 = maskformerdecoder_forward(hidden_dim)

        self.deconv1=deconv(hidden_dim)
        self.deconv2=deconv(hidden_dim)
        self.deconv3 = deconv(hidden_dim)

    def forward(self, transformer_middle, self_mask=None, cross_mask=None):

        key12=transformer_middle[3]
        key9=transformer_middle[2]
        key6= transformer_middle[1]
        
        key9=self.deconv1(key9)
        key6=self.deconv2(key6)
        key6=self.deconv3(key6)

        level_embedding12=self.level_embedding.weight[0][None,None,:]
        level_embedding9=self.level_embedding.weight[1][None,None,:]
        level_embedding6 = self.level_embedding.weight[2][None, None, :]
        
        src12,ket_po12=self.process12(key12,level_embedding12)
        src9, ket_po9 = self.process9(key9,  level_embedding9)
        src6, ket_po6 = self.process6(key6, level_embedding6)

        input=transformer_middle[0]
        query_embedding = self.query_embedding.weight.unsqueeze(0).expand(input.shape[0], -1, -1)
        query_feat=self.query_feat.weight.unsqueeze(0).expand(input.shape[0], -1, -1)

        batch_size, sequence_length, hidden_dim = input.size(0), input.size(1), input.size(2)
        query=query_feat
        
        i=0
        for layer in self.layers:
            if i%3==0:
                query = layer(query,query_embedding,src12,ket_po12)#这里只需要输入key或者value作为memory输入transformer decoder layers
            if i%3==1:
                query = layer(query, query_embedding, src9, ket_po9)
            if i%3==2:
                query = layer(query, query_embedding, src6, ket_po6)
            i+=1
            
        return query


