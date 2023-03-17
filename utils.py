# some utils for the project

import os
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


def load_data(file, target_dir, test_num):
    '''
    file: list of file names (for ct, mri)
    target_dir: file directory
    test_num: number of test data
    return: torch .pt file store ct and mri
    '''

    test_ind = np.random.choice(len(file), size=test_num, replace = False)
    print(test_ind)
    test = []
    for ind in test_ind:
        test.append(file[ind])
    
    #print(test)
    
    HEIGHT = 256
    WIDTH = 256

    # 1 channel image, with shape 256x256
    data_ct = torch.empty(0, 1, HEIGHT, WIDTH)
    data_mri = torch.empty(0, 1, HEIGHT, WIDTH)
    data_ct_t = torch.empty(0, 1, HEIGHT, WIDTH)
    data_mri_t = torch.empty(0, 1, HEIGHT, WIDTH)
    
    for f in file:
        # read data and normalize, change this
        img_ct = io.imread(os.path.join(target_dir, "CT", f)).astype(np.float32) / 255.
        img_mri = io.imread(os.path.join(target_dir, "MRI", f)).astype(np.float32) / 255.
        img_ct = torch.from_numpy(img_ct)
        img_mri = torch.from_numpy(img_mri)
        img_ct = img_ct.unsqueeze(0).unsqueeze(0) # change shape to (1, 1, 256, 256)
        img_mri = img_mri.unsqueeze(0).unsqueeze(0)

        if f not in test:
            data_ct = torch.cat((data_ct, img_ct), dim = 0)
            data_mri = torch.cat((data_mri, img_mri), dim = 0)
        else:
            data_ct_t = torch.cat((data_ct_t, img_ct), dim = 0)
            data_mri_t = torch.cat((data_mri_t, img_mri), dim = 0)
    
    return data_ct, data_mri, data_ct_t, data_mri_t