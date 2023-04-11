# breath-wise cross spatial attention module

'''
Author: Simon Zhou

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from breath_wise_cross_att import BreathWise


class ChannelGate(nn.Module):
    # channel gate for cross spatial attention
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels,
                      gate_channels // reduction_ratio), nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

        self.shared_conv = nn.Sequential(
            nn.Conv2d(gate_channels,gate_channels//reduction_ratio,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(gate_channels//reduction_ratio,gate_channels,1,bias=False)
            )

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.conv_S = nn.Sequential(
        #     #nn.MaxPool2d(2),
        #     nn.Conv2d(channel_S, channel_S, kernel_size = 1, stride=2, bias=False),
        #     nn.BatchNorm2d(channel_S),
        #     nn.ReLU()
        # ) # out of this is c*h*w

        # self.conv_Y = nn.Sequential(
        #     nn.Conv2d(channel_Y, channel_S, 1, bias=False),
        #     nn.BatchNorm2d(channel_S),
        #     nn.ReLU()
        # ) # reduce channel of y, the output is c*h*w
        
    def forward(self, x, y):
        # x: from enc, y: from dec that already up sampled
        assert x.shape == y.shape, "require x and y are in the same shape!"
        # x = self.conv_S(x) # c*h*w
        # print(x.shape)
        # y = self.conv_Y(y)
        # print(y.shape)
    
        channel_att_sum = None
        
        # glocal information from deeper-level y
        # ave_y = F.avg_pool2d(y, (y.size(2), y.size(3)), stride=(y.size(2), y.size(3)))
        # channel_att_y = self.mlp(ave_y.view(ave_y.size(0), -1))
        
        ave_y = self.avgpool(y)
        channel_att_y = self.shared_conv(ave_y)
        
        # lobal information from shallow-level x
        # max_s = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # channel_att_s = self.mlp(max_s.view(max_s.size(0), -1))
        max_s = self.maxpool(x)
        channel_att_s = self.shared_conv(max_s)
        
        channel_att_sum = channel_att_y + channel_att_s
        
        scale = F.sigmoid(channel_att_sum)#.unsqueeze(2).unsqueeze(3).expand_as(x)
        #print(scale.shape)
        
        return x * scale

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out, ks=3):
        super(single_conv, self).__init__()

        self.conv = nn.Conv2d(ch_in,
                              ch_out,
                              kernel_size=ks,
                              stride=1,
                              padding=(ks - 1) // 2,
                              bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x, y):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(y, 1).unsqueeze(1)),
            dim=1)
 

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = single_conv(2, 1, ks = kernel_size)

    def forward(self, x, y):
        x_compress = self.compress(x, y)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale
    

class BreathWiseCrossSpatialAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(BreathWiseCrossSpatialAttention, self).__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.spatial_gate = SpatialGate()
        self.bw = BreathWise(gate_channels, gate_channels)
        #self.ln = nn.BatchNorm2d(gate_channels)
    def forward(self, x, y):
        # x: S, y: Y
        out = self.channel_gate(x, y)
        out = self.spatial_gate(out, y)
        
        out = out + x
        #out = self.ln(out)
        
        att = self.bw(out)
        return att + out
