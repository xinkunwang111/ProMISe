# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE:  https://github.com/facebookresearch/mae/blob/main/models_mae
# ViT:  https://github.com/facebookresearch/mae/blob/main/models_vit
# --------------------------------------------------------
#models_vit
from functools import partial

import torch
import torch.nn as nn
import numpy as np

import timm.models.vision_transformer

from timm.models.vision_transformer import PatchEmbed, Block

# from util.pos_embed import get_2d_sincos_pos_embed
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, args=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.args = args
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.embed_dim = kwargs['embed_dim']
        self.initialize_weights()
        
        #---------------------------------------------------------------------------
        print(self.embed_dim)
        
        num_patches = 4096 # （1024除以16）**2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, self.embed_dim), requires_grad=True) # pos_embed 这里原本requires_grad=False（mae代码里面）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))# cls_token
        


        if_pretrain =False
        hp1=1.0
       

        
        if  if_pretrain == False:
            if  hp1 == 1.0:
                pass
            else:
                self.size1 = int(round(self.embed_dim*hp1)) #self.args.
                if self.args.else_part == "main_part":
                    
                    self.head = nn.Linear(self.size1, kwargs['num_classes']) 
                else:
                    
                    self.head = nn.Linear(self.embed_dim-self.size1, kwargs['num_classes']) # kwargs['num_classes']

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):#-------------------------------------------------------------------------
        B = x.shape[0]
        print( 'x.shape', x.shape) 
        
        
        input_tensor = x
        target_shape = [8, 4096, 384]
        padding_columns = target_shape[2] - input_tensor.shape[2]
        padding = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], padding_columns, device=input_tensor.device)
        output_tensor = torch.cat([input_tensor, padding], dim=2)
        x=output_tensor
        print( 'x.shape', x.shape) 

        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        print("cls_tokens.shape", cls_tokens.shape) 
        x = torch.cat((cls_tokens, x), dim=1)
        print("after cls_tokens x.shape", x.shape) 
        x = x + self.pos_embed
        x = self.pos_drop(x)

        
        for blk in self.blocks:
            x = blk(x)
            

        if self.global_pool: 
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        print('outcome shape:',outcome.shape) 
        return outcome


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) #384
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model







