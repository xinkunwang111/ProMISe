# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .featuremap import FeatureMap12,FeatureMap12_9,FeatureMap12_9_6, FeatureMap12_9_6_3, concat, FeatureScale
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import numpy as np
import cv2
import os
from .transformer_decoder import *
from .common import *

from .models_vit import vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14

def init_point_sampling(mask, get_point=1):
# from
# https://github.com/OpenGVLab/SAM-Med2D/blob/main/utils.py 

    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu()
        mask = mask.numpy()

    # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                              dtype=torch.int)
        return coords, labels



class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        args = None

    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        
        self.args = args
        
        self.GT = "GT" in self.args.model_name
        self.use_resnet = "resnet" in self.args.model_name
        self.use_cross = "cross" in self.args.model_name
        self.APM = "APM" in self.args.model_name
        self.IPS = "IPS" in self.args.model_name
        self.point_num = self.args.point_num

        if self.APM and not self.IPS and not self.GT:
            self.point_num = self.args.point_num
        else:
          self.ips_global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
          self.ips_ffn = FFNBlock(256, 4 * 256, 4 * 256)

        if self.use_cross and not self.GT:
            self.apm_cross= MaskformerDecoder(1,768,12,self.args.point_num)
            self.apm_cross_prompt=nn.Linear(768,2)

        if self.use_resnet and not self.GT:
            self.apm_feature_map = Sam.get_feature()

            self.apm_resnet = Sam.get_backbone(point_num = self.args.point_num)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @staticmethod
    def get_backbone(backbone_name='resnet34', args = None,point_num=1):
        return {'resnet18': ResNet18(point_num=point_num),
                'resnet34': ResNet34(point_num=point_num),
                'resnet50': ResNet50(point_num=point_num),
                'resnet101': ResNet101(point_num=point_num),
                'resnet152': ResNet152(point_num=point_num),
                'vit_small_patch16': vit_small_patch16(args=args, num_classes=32)}[backbone_name]

    @staticmethod
    def get_feature(name='FeatureMap12_9_6', args = None):#-----------124line改concat
        return {'FeatureMap12': FeatureMap12(args=args),
                'FeatureMap12_9': FeatureMap12_9(args=args),
                'FeatureMap12_9_6': FeatureMap12_9_6(args=args),
                'FeatureMap12_9_6_3': FeatureMap12_9_6_3(args=args)}[name]

    #@torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        #select 
        if self.use_cross:
            if self.APM and not self.IPS:
              input_images = batched_input["image"]
              image_embeddings,transformer_middle = self.image_encoder(input_images)
              
              output=self.apm_cross(transformer_middle)
              prompt = self.apm_cross_prompt(output)
              prompt=torch.sigmoid(prompt)
              prompt=prompt * 1024
              point_num = self.args.point_num

            if self.IPS and self.GT:
              input_images = batched_input["image"]
              image_embeddings,transformer_middle = self.image_encoder(input_images)

              point_num = self.args.point_num

              prompt = torch.ones((image_embeddings.shape[0], point_num, 2), device='cuda:0')#ips use gt prompt,here the prompt variable has no significance
              decoder_token = self.ips_global_avg_pool(image_embeddings)
              decoder_token = decoder_token.flatten(1)
              decoder_token = self.ips_ffn(decoder_token)
              decoder_token = decoder_token.view(decoder_token.shape[0], -1, 256)

            if self.APM and self.IPS and not self.GT:
                input_images = batched_input["image"]
                image_embeddings,transformer_middle = self.image_encoder(input_images)
                
                output=self.apm_cross(transformer_middle)
                prompt = self.apm_cross_prompt(output)
                prompt=torch.sigmoid(prompt)
                prompt=prompt * 1024
                point_num = self.args.point_num

                decoder_token = self.ips_global_avg_pool(image_embeddings)
                decoder_token = decoder_token.flatten(1)
                decoder_token = self.ips_ffn(decoder_token)
                decoder_token = decoder_token.view(decoder_token.shape[0], -1, 256)
                
        if self.use_resnet:
            if self.APM and not self.IPS:
                input_images = batched_input["image"]

                image_embeddings,output_feature = self.image_encoder(input_images)

                concat_feature = concat(output_feature, '6_9_12')
                image_embeddings_resnet = self.apm_feature_map(concat_feature)
                output, learnable_token_decoder = self.apm_resnet(image_embeddings_resnet)
                prompt = torch.sigmoid(output)
                prompt = prompt * 1024
                point_num = self.args.point_num

            if self.IPS and self.GT:
                input_images = batched_input["image"]
                image_embeddings, transformer_middle = self.image_encoder(input_images)

                point_num = self.args.point_num

                prompt = torch.ones((image_embeddings.shape[0], point_num, 2),
                                    device='cuda:0')  # ips use gt prompt,here the prompt variable has no significance
                decoder_token = self.ips_global_avg_pool(image_embeddings)
                decoder_token = decoder_token.flatten(1)
                decoder_token = self.ips_ffn(decoder_token)
                decoder_token = decoder_token.view(decoder_token.shape[0], -1, 256)

            if self.APM and self.IPS and not self.GT:
                input_images = batched_input["image"]

                image_embeddings,output_feature = self.image_encoder(input_images)

                concat_feature = concat(output_feature, '6_9_12')
                image_embeddings_resnet = self.apm_feature_map(concat_feature)
                output, learnable_token_decoder = self.apm_resnet(image_embeddings_resnet)
                prompt = torch.sigmoid(output)
                prompt = prompt * 1024

                point_num = self.args.point_num

                decoder_token = self.ips_global_avg_pool(image_embeddings)
                decoder_token = decoder_token.flatten(1)
                decoder_token = self.ips_ffn(decoder_token)
                decoder_token = decoder_token.view(decoder_token.shape[0], -1, 256)


        labels_torch = batched_input["labels_torch"]
        batched_input["prompts"] = []
      
        for nn in range(prompt.shape[0]):#batch_size=8，
            batched_input["prompts"].append({"point_coords":prompt[nn,:].reshape(1, self.args.point_num, 2), "point_labels":labels_torch.reshape(1, labels_torch.shape[0])})

        outputs = []
        
        i=0
        j = 0

        for image_record, curr_embedding in zip(batched_input["prompts"], image_embeddings):
            if self.IPS and self.GT and not self.APM:
              decoder_token_ = decoder_token[i,:,:]
              i+=1
              true_masks = batched_input["true_masks"][j][0]
              j+=1
              
              points_coords, point_labels = init_point_sampling(true_masks, get_point=point_num)
              points_coords = points_coords.unsqueeze(0).to("cuda:0")
              point_labels = point_labels.unsqueeze(0).to("cuda:0")
              points = (points_coords, point_labels)
              sparse_embeddings, dense_embeddings = self.prompt_encoder(
                  points=points,
                  boxes=None,
                  masks=image_record.get("mask_inputs", None)
              )

              low_res_masks, iou_predictions = self.mask_decoder(
                  image_embeddings=curr_embedding.unsqueeze(0),
                  image_pe=self.prompt_encoder.get_dense_pe(),
                  sparse_prompt_embeddings=sparse_embeddings,
                  dense_prompt_embeddings=dense_embeddings,
                  multimask_output=multimask_output,
                  decoder_token = decoder_token_)

            if self.IPS and self.APM and not self.GT:
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])

                else:
                    points = None
                    
                decoder_token_ = decoder_token[i, :, :]
                i += 1
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=points,                  
                        boxes=None,
                        masks=image_record.get("mask_inputs", None)
                    )

                low_res_masks, iou_predictions = self.mask_decoder(
                        image_embeddings=curr_embedding.unsqueeze(0),
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                        decoder_token=decoder_token_)
            if self.IPS and self.APM and self.GT:
                decoder_token_ = decoder_token[i, :, :]
                i += 1
                true_masks = batched_input["true_masks"][j][0]
                j += 1
                
                points_coords, point_labels = init_point_sampling(true_masks, get_point=point_num)
                points_coords = points_coords.unsqueeze(0).to("cuda:0")
                point_labels = point_labels.unsqueeze(0).to("cuda:0")
                points = (points_coords, point_labels)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=image_record.get("mask_inputs", None)
                )

                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                    decoder_token=decoder_token_)

            if self.APM and not self.IPS and not self.GT:
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])
                else:
                    points = None

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                  points=points,
                  boxes=None,
                  masks=image_record.get("mask_inputs", None)
                )
              
                low_res_masks, iou_predictions = self.mask_decoder(
                      image_embeddings=curr_embedding.unsqueeze(0),
                      image_pe=self.prompt_encoder.get_dense_pe(),
                      sparse_prompt_embeddings=sparse_embeddings,
                      dense_prompt_embeddings=dense_embeddings,
                      multimask_output=multimask_output,
                    decoder_token = None,
                )

            low_res_masks = torch.sigmoid(low_res_masks)
            masks = low_res_masks
            
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "pred_prompts": batched_input["prompts"],
                    "box_coords":prompt[0]
                }
            )
            #print("---------------------------------------")
        middle=[]
        return outputs,middle

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
