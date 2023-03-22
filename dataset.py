from torch.utils.data import Dataset
import torch

class BraTS_2d(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.load_data()
    
    def load_data(self):
        if self.mode == 'train':
            self.imgs = torch.load(self.data_dir + 'train_img.pt') / 255.
            self.masks = torch.load(self.data_dir + 'train_mask.pt')
            # spilit train and val
            self.imgs = self.imgs[:self.imgs.shape[0] * 8 // 10]
            self.masks = self.masks[:self.masks.shape[0] * 8 // 10]
        elif self.mode == 'val':
            self.imgs = torch.load(self.data_dir + 'train_img.pt') / 255.
            self.masks = torch.load(self.data_dir + 'train_mask.pt')
            # spilit train and val
            self.imgs = self.imgs[self.imgs.shape[0] * 8 // 10:]
            self.masks = self.masks[self.masks.shape[0] * 8 // 10:]
        elif self.mode == 'test':
            self.imgs = torch.load(self.data_dir + 'test_img.pt') / 255.
            self.masks = torch.load(self.data_dir + 'test_mask.pt')
        else:
            raise ValueError('mode should be train or test')
        
    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask
    
