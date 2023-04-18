# this implements transBTS model, ref: https://arxiv.org/abs/2103.04430
# but in 2D

# Author: Simon

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(net, init_type='normal', gain=0.02):
    # initialize network weights
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Swish(nn.Module):
    # swish activation function
    def forward(self, x):
        return x * torch.sigmoid(x)


class Norm_layer(nn.Module):
    # collections of normalization layers
    def __init__(self, channels, type="batch"):
        super().__init__()
        self.type = type

        if type == "group":
            self.norm = nn.GroupNorm(num_groups=8,
                                     num_channels=channels)
        elif type == "batch":
            self.norm = nn.BatchNorm2d(channels)
        elif type == "instance":
            self.norm = nn.InstanceNorm2d(channels)
        elif type == "layer":
            self.norm = nn.LayerNorm(channels, eps=1e-5)
        else:
            raise NotImplementedError(
                "normalization layer [%s] is not implemented!" % type)

    def forward(self, x):
        return self.norm(x)


class InitConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        y = self.conv(x)
        y = self.drop(y)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm="batch"):
        super(EnBlock, self).__init__()

        self.norm = norm
        self.bn1 = Norm_layer(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = Norm_layer(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        # residule connection
        y = y + x

        return y


class EnDown(nn.Module):
    # downsample by strided convolutions with stride 2, instead of maxpooling
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y


class proj_back(nn.Module):
    # project back to original number of channels
    def __init__(self, in_channels, out_channels):
        super(proj_back, self).__init__()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # 512 is the number of channels output by the encoder
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1

class up_conv_init(nn.Module):
    def __init__(self, in_channels):
        super(up_conv_init, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        # residual connection
        x1 = x1 + x

        return x1


class up_conv(nn.Module):
    # upsample convolutions in decoder
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(ch_out*2, ch_out, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.up(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        y = self.bn1(y)
        y = self.relu(y)
        return y
    

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        # residual connection
        x1 = x1 + x

        return x1

class Encoder(nn.Module):
    # code ref: https://github.com/LeeJunHyun/Image_Segmentation/blob/db34de21767859e035aee143c59954fa0d94bbcd/network.py
    def __init__(self):
        super(Encoder, self).__init__()

        self.init1 = InitConv(1, 32)
        self.conv1 = EnBlock(32, 32)
        self.down1 = EnDown(32, 64)
        
        self.conv2 = EnBlock(64, 64) 
        self.down2 = EnDown(64, 128)
        
        self.conv3 = EnBlock(128, 128)
        self.down3 = EnDown(128, 256)
        
        self.conv4 = EnBlock(256, 256)
        self.down4 = EnDown(256, 512)
        
        self.conv5 = EnBlock(512, 512)  

    def forward(self, input):
        x = self.init1(input)
        x = self.conv1(x)
        # skip-connection 1
        concat1 = x # b, 32, 128, 128
        
        x = self.down1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        # skip-connection 2
        concat2 = x # b, 64, 64, 64
        
        x = self.down2(x)
        x = self.conv3(x)
        x = self.conv3(x)
        # skip-connection 3
        concat3 = x # b, 128, 32, 32
        
        x = self.down3(x)
        x = self.conv4(x)
        x = self.conv4(x)
        # skip-connection 4
        concat4 = x # b, 256, 16, 16
        
        x = self.down4(x)
        x = self.conv5(x) 
        x = self.conv5(x) # x should be b,512,8,8 in latent space
        
        # return concat layers for connecting to decoder
        return x, (concat1, concat2, concat3, concat4)


class Transformer(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=1024,
                 nhead=8,
                 dim_feedforward=2048,
                 num_pixel=64,
                 dropout=0.1):

        super(Transformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model,
                                                            nhead,
                                                            dim_feedforward,
                                                            dropout,
                                                            batch_first=True,
                                                            norm_first=True)

        self.transformer = nn.TransformerEncoder(self.transformer_layer,
                                                 num_layers)

        # learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_pixel, d_model))
        
        # linear proj
        self.linear_proj = nn.Linear(d_model, d_model)
        
        # pe_dropout
        self.pe_drop = nn.Dropout(p=0.1)
        
        # post norm
        self.post_norm = nn.LayerNorm(1024)
        
        # conv_proj
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv_proj = nn.Conv2d(512, d_model, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #print("pre x.shape", x.shape)
        # x: [B, C, H, W]
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_proj(x)
        B = x.shape[0]
        C = x.shape[1]
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # [B, H*W, C]
        #  pe
        x = x + self.pos_embedding
        x = self.pe_drop(x)
        #print("post x.shape", x.shape)
        x = self.transformer(x)
        x = self.post_norm(x)
        return x


class SegmentorBTS(nn.Module):
    def __init__(self, num_cls):
        # segmentation head to produce segmentation map for cross attention module
        super(SegmentorBTS, self).__init__()

        self.conv_proj_back = proj_back(1024, 512)
        self.conv_again = up_conv_init(512)
        
        # 8 ->16
        self.up5 = up_conv(512, 256)
        self.upconv5 = DeBlock(256)

        # 16 -> 32
        self.up4 = up_conv(256, 128)
        self.upconv4 = DeBlock(128)

        # 32-> 64
        self.up3 = up_conv(128, 64)
        self.upconv3 = DeBlock(64)

        # 64 -> 128
        self.up2 = up_conv(64, 32)
        self.upconv2 = DeBlock(32)

        # 64 channel to 1
        self.final_conv = nn.Conv2d(32,
                                    num_cls,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.seg_act = nn.Sigmoid()

    def forward(self, lat_feat, concat_feats):
        # channels: 64, 128, 256, 512
        lat_feat = self.conv_proj_back(lat_feat)
        lat_feat = self.conv_again(lat_feat)
        concat1, concat2, concat3, concat4 = concat_feats
        # lat_feat: 512,8,8
        
        #print(cross1.shape, lat_feat.shape)
        x = self.up5(lat_feat, concat4)
        x = self.upconv5(x) #self.upconv5(torch.cat([cross1, x], dim=1)) # should be 256, 16, 16
        #print(x.shape)

        #print("cross2 shape", cross2.shape)
        x = self.up4(x, concat3)
        x = self.upconv4(x) #self.upconv4(torch.cat([cross2, x], dim=1)) # should be 128, 32, 32
        #print(x.shape)

        #print("cross3 shape", cross3.shape)
        x = self.up3(x, concat2)
        x = self.upconv3(x) #(torch.cat([cross3, x], dim=1)) # should be 64, 64, 64
        #print(x.shape)
        
        #print("cross4 shape", cross4.shape)
        x = self.up2(x, concat1)
        x = self.upconv2(x) #(torch.cat([cross4, x], dim=1)) # should be 32, 128, 128
        #print(x.shape)

        x = self.final_conv(x) # should be 1, 128, 128

        # HANDLE this when calculating loss
        # segmentation map, sigmoid maps to [0,1]
        #x = F.sigmoid(x)  # TODO: check if sigmoid is needed

        return x


class TransBTS(nn.Module):
    # Multi-task Transformer U-Net

    def __init__(self, num_cls=1): #TODO: check aross_att
        super(TransBTS, self).__init__()
        self.encoder = Encoder()
        self.transformer = Transformer()
        # whether we want to use cross attention module
        self.segmentor = SegmentorBTS(num_cls)

    def forward(self, x):
        lat_feat, concat_feats = self.encoder(x)
        B, C, H, W  = lat_feat.shape
        lat_feat = self.transformer(lat_feat)
        lat_feat = lat_feat.transpose(1, 2).contiguous().view(B, -1, H, W)
        seg_map = self.segmentor(lat_feat, concat_feats)

        rec_img = torch.zeros_like(seg_map)
        return seg_map, rec_img