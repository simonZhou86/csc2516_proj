# some utils for the project

import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split, Dataset
'''
@ author: Simon Zhou

'''


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def normalize(im):
	# normalize to [-1,1]
	mins = [im[idx].min() for idx in range(len(im))]
	maxes = [im[idx].max() for idx in range(len(im))]
	
	for idx in range(len(im)):
		min_val = mins[idx]
		max_val = maxes[idx]
		if min_val == max_val:
			im[idx] = torch.zeros(im[idx].shape)
		else:
			im[idx] = 2*(im[idx] - min_val)/(max_val - min_val)-1
   

class getIndex(Dataset):
    # might be useful for creating dataloader
	def __init__(self, total_len):
		self.tmp = total_len
		self.total_len = self.tmp
	
	def __len__(self):
		return self.total_len
	
	def __getitem__(self, ind):
		return torch.Tensor([ind])

