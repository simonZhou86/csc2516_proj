# breath wise cross attention
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

'''
Author: Simon
Breath-wise cross attention, attention learned locally and dialtion leanred with a larger receptive field
'''


class BreathWise(nn.Module):
    # Breath-Wise convolution block through dilations
    # used in the breath-wise cross attention
    
    def __init__(self, in_channel, out_channel, activation='relu'):
        super(BreathWise, self).__init__()
        
        self.d_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = "same")
        self.d_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = "same", dilation=3)
        self.d_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = "same", dilation=5)
        self.final_conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        if activation == 'gelu':
            self.actf = nn.GELU()
        elif activation == "relu":
            self.actf = nn.ReLU()
        elif activation == "lrelu":
            self.actf = nn.LeakyReLU()
        else:
            raise NotImplementedError
        
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm2d(out_channel)
        
    
    def forward(self, x):
        # x should be output from the cross attention (but before concat with the encoder output)
        x1 = self.d_conv1(x)
        x1 = self.actf(x1)
        x1 = self.dropout(x1)
        #print("x1: ", x1.shape)

        x2 = self.d_conv2(x)
        x2 = self.actf(x2)
        x2 = self.dropout(x2)
        #print("x2: ", x2.shape)

        x3 = self.d_conv3(x)
        x3 = self.actf(x3)
        x3  = self.dropout(x3)
        #print("x3: ", x3.shape)
        x_all = x1 + x2 + x3 # element wise add
        x_all = self.final_conv(x_all)
        x_all = self.bn(x_all)
        x_all = self.actf(x_all)
        
        return x_all        
    
    
class BreathWiseCrossAttention(nn.Module):
    # breath wise cross attention
        
    def __init__(self, embed_dim: int, num_heads: int, channel_S: int, channel_Y: int, bias=False) -> None:
        super(BreathWiseCrossAttention, self).__init__()
        # S: previous layer encoder output
        # Y: current layer decoder output
        # S: should be c*2h*2w, and Y should be 2c*h*w
        self.conv_S = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(channel_S, channel_S, kernel_size = 1, stride=2, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        ) # out of this is c*h*w

        self.conv_Y = nn.Sequential(
            nn.Conv2d(channel_Y, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        ) # reduce channel of y, the output is c*h*w

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True) # multi head attention, output is c*h*w

        self.upsample_weights = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),#.apply(lambda m: nn.init.xavier_uniform_(m.weight.data)),
            nn.BatchNorm2d(channel_S),
            nn.Sigmoid(),
        ) # upsample to obtain the weights for element-wise product
        
        self.conventional_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU(),
        ) # normal upsample block
        
        self.bw_block = BreathWise(channel_S, channel_S) # breath-wise block
        
    
    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s_enc = s
        s = self.conv_S(s)
        y = self.conv_Y(y)
        y_orig = y

        b, c, h, w = s.size()
        s = s.permute(0, 2, 3, 1).view((b, h * w, c)) # b, h*w, c

        b, c, h, w = y.size()
        y = y.permute(0, 2, 3, 1).view((b, h * w, c)) # b, h*w, c

        y, _ = self.mha(y, y, s, need_weights=False) # b, h*w, c
        y = y.view((b, h, w, c)).permute(0, 3, 1, 2) # b, c, h, w
        
        
        y = self.upsample_weights(y) # attention weights
        
        y = torch.mul(y, s_enc) # get attention features
        
        #print(y.shape, y_orig.shape) # debug, check the shape
        #print(y.shape)
        # whether we need residual for s + y?
        y = self.bw_block(y + s_enc) # breath-wise block
        
        return y
