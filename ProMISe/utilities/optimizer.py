import torch.nn as nn
import torch.optim as optim

def build_optimizer(optimizer_type, model_conv, INITIAL_LR):
	if optimizer_type == "Adam":
		optimizer = optim.Adam(model_conv.parameters(), lr=INITIAL_LR)
		return optimizer
	elif optimizer_type == "AdamW":
		optimizer = optim.AdamW(model_conv.parameters(), lr=INITIAL_LR)
		return optimizer
	elif optimizer_type == "SGD":
		optimizer = optim.SGD(model_conv.parameters(), lr=INITIAL_LR)
		return optimizer
	