import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

'''
Encoder first, no attention at this point, use strided convolutions to downsample, encode to latent space

latent space feed to transformer, model long-term dependencies, global attention

Decoder has local attention, attention is after the skip connections at each level, use upsample and convolutions to upsample, decode to segmentation mask


'''

def init_weights(net, init_type='normal', gain=0.02):
    # initialize network weights
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    # general conv block, 2 x (conv layers, batchnorm, relu)
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):
    # downsample by strided convolutions with stride 2, instead of maxpooling
    def __init__(self,ch_in,ch_out):
        super(down_conv).__init__()
    
        self.down = nn.Sequential(nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=2,padding=1,bias=True),
                                nn.BatchNorm2d(ch_out),
                                nn.ReLU(inplace=True))
    
    def forward(self,x):
        x = self.down(x)
        return x
    
    
class up_conv(nn.Module):
    # upsample convolutions in decoder
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    # ref: https://arxiv.org/abs/1804.03999
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class Encoder(nn.Module):
    # https://github.com/LeeJunHyun/Image_Segmentation/blob/db34de21767859e035aee143c59954fa0d94bbcd/network.py
    def __init__(self):
        super(Encoder).__init__()
        
        self.conv1 = conv_block(1,64)
        self.down1 = down_conv(64,64) # 128->64
        self.conv2 = conv_block(64,128)
        self.down2 = down_conv(128,128) # 64->32
        self.conv3 = conv_block(128,256)
        self.down3 = down_conv(256,256) # 32->16
        self.conv4 = conv_block(256,512)
        self.down4 = down_conv(512,512) # 16-> 8
        self.conv5 = conv_block(512,1024)
    
    def forward(self, input):
        x = self.conv1(input)
        # skip-connection 1
        concat1 = x
        x = self.down1(x)
        x = self.conv2(x)
        # skip-connection 2
        concat2 = x
        x = self.down2(x)
        x = self.conv3(x)
        # skip-connection 3
        concat3 = x
        x = self.down3(x)
        x = self.conv4(x)
        # skip-connection 4
        concat4 = x
        x = self.down4(x)
        x = self.conv5(x) # x should be 1024x8x8 in latent space
        
        return x, (concat1, concat2, concat3, concat4)



class auxiliaryDecoder(nn.Module):
    def __init__(self):
        super(auxiliaryDecoder).__init__()

        # 8 ->16
        self.up1 = up_conv(1024,512)
        self.upconv1 = conv_block(512,512)
        # 16 -> 32
        self.up2 = up_conv(512,256)
        self.upconv2 = conv_block(256,256)
        # 32-> 64
        self.up3 = up_conv(256,128)
        self.upconv3 = conv_block(128,128)
        # 64 -> 128
        self.up4 = up_conv(64,64)
        self.upconv4 = conv_block(64,64)
        
        # 64 channel to 1
        self.final_conv = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0,bias=True)
        
    
    def forward(self, lat_fea):
        x = self.up1(lat_fea)
        x = self.upconv1(x)
        
        x = self.up2(x)
        x = self.upconv2(x)
        
        x = self.up3(x)
        x = self.upconv3(x)
        
        x = self.up4(x)
        x = self.upconv4(x)
        
        x = self.final_conv(x)
        
        x = F.tanh(x) # map to [-1,1], depands on the range of input image, use sigmoid if input image is in [0,1]
        
        return x