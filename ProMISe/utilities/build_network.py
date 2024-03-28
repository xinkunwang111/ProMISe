import torch
import torch.nn as nn

def build_network(model_name,args, pretrained = False, snapshot = None, num_classes = 3, vis = None):

	print(model_name, "loaded")

	if "SAM" in model_name:
		from functools import partial
		from utilities.Networks.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

		def build_sam_vit_h(checkpoint=None):
			return _build_sam(
				encoder_embed_dim=1280,
				encoder_depth=32,
				encoder_num_heads=16,
				encoder_global_attn_indexes=[7, 15, 23, 31],
				checkpoint=checkpoint,
				args = args,
			)
		
		def build_sam_vit_l(checkpoint=None):
			return _build_sam(
				encoder_embed_dim=1024,
				encoder_depth=24,
				encoder_num_heads=16,
				encoder_global_attn_indexes=[5, 11, 17, 23],
				checkpoint=checkpoint,
				args = args,
			)
		
		
		def build_sam_vit_b(checkpoint=None):
			return _build_sam(
				encoder_embed_dim=768,
				encoder_depth=12,
				encoder_num_heads=12,
				encoder_global_attn_indexes=[2, 5, 8, 11],
				checkpoint=checkpoint,
				args = args,
			)

		def _build_sam(
			encoder_embed_dim,
			encoder_depth,
			encoder_num_heads,
			encoder_global_attn_indexes,
			checkpoint=None,
			args = None,
		):
			prompt_embed_dim = 256
			image_size = 1024 #384 1024
			vit_patch_size = 16
			image_embedding_size = image_size // vit_patch_size
			sam = Sam(
				image_encoder=ImageEncoderViT(
					depth=encoder_depth,
					embed_dim=encoder_embed_dim,
					img_size=image_size,
					mlp_ratio=4,
					norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
					num_heads=encoder_num_heads,
					patch_size=vit_patch_size,
					qkv_bias=True,
					use_rel_pos=True,
					global_attn_indexes=encoder_global_attn_indexes,
					window_size=14,
					out_chans=prompt_embed_dim,
					args = args,
				),
				prompt_encoder=PromptEncoder(
					embed_dim=prompt_embed_dim,
					image_embedding_size=(image_embedding_size, image_embedding_size),
					input_image_size=(image_size, image_size),
					mask_in_chans=16,
				),
				mask_decoder=MaskDecoder(
					num_multimask_outputs=3,
					transformer=TwoWayTransformer(
						depth=2,
						embedding_dim=prompt_embed_dim,
						mlp_dim=2048,
						num_heads=8,
					),
					transformer_dim=prompt_embed_dim,
					iou_head_depth=3,
					iou_head_hidden_dim=256,
					args = args,

				),
				pixel_mean=[123.675, 116.28, 103.53],
				pixel_std=[58.395, 57.12, 57.375],
				args = args,
			)
			
			if checkpoint is not None:
				with open(checkpoint, "rb") as f:
					state_dict = torch.load(f)
				sam.load_state_dict(state_dict, strict=False)
			
			return sam

		if "SAM_vit-h" in model_name:
			model_conv = build_sam_vit_h(checkpoint = args.check_point_path)
		elif "SAM_vit-l" in model_name:
			model_conv = build_sam_vit_l(checkpoint = args.check_point_path)
		elif "SAM_vit-b" in model_name:
			model_conv = build_sam_vit_b(checkpoint = args.check_point_path)

		return model_conv








