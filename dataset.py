# dataset class, load data and transform data

# Author: Frank Liu
# Last modify: Simon Zhou Mar 22, 2023

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

class BraTS_2d(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        # add transformation method
        self.need_transform = True
        if self.transform is None:
            self.need_transform = False
        elif self.transform == "resize":
            self.resize = transforms.Resize((128, 128)) # or we can directly use F.interpolate instead
        else:
            raise NotImplementedError
        self.load_data()
    
    def load_data(self):
        if self.mode == 'train':
            self.imgs = torch.load(self.data_dir + 'train_img.pt')
            self.masks = torch.load(self.data_dir + 'train_mask.pt')
            # spilit train and val
            self.imgs = self.imgs[:self.imgs.shape[0] * 8 // 10]
            self.masks = self.masks[:self.masks.shape[0] * 8 // 10]
        elif self.mode == 'val':
            self.imgs = torch.load(self.data_dir + 'train_img.pt')
            self.masks = torch.load(self.data_dir + 'train_mask.pt')
            # spilit train and val
            self.imgs = self.imgs[self.imgs.shape[0] * 8 // 10:]
            self.masks = self.masks[self.masks.shape[0] * 8 // 10:]
        elif self.mode == 'test':
            self.imgs = torch.load(self.data_dir + 'test_img.pt')
            self.masks = torch.load(self.data_dir + 'test_mask.pt')
        else:
            raise ValueError('mode should be train or test')
        
    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        # resize is the only valid option for now
        if self.need_transform:
            # img, mask should be N,C,H,W
            img = self.resize(img)
            mask = self.resize(mask)
        return img, mask
    
