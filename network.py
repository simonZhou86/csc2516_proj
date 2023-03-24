import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from breath_wise_cross_att import *
'''
@ author: Simon Zhou, Frank Liu

Encoder first, no attention at this point, use strided convolutions to downsample, encode to latent space, decouple with the decoder

latent space feed to transformer, model long-term dependencies, global attention

Decoder has local attention, attention is after the skip connections at each level, use upsample and convolutions to upsample, decode to segmentation mask

auxiliary decoder: reconstruct the input image
'''


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


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32,
                               num_channels=channels,
                               eps=1e-6,
                               affine=True)

    def forward(self, x):
        return self.gn(x)


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
            self.norm = nn.GroupNorm(num_groups=32,
                                     num_channels=channels,
                                     eps=1e-6,
                                     affine=True)
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


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out, ks, use_relu=False):
        super(single_conv, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(ch_in,
                              ch_out,
                              kernel_size=ks,
                              stride=1,
                              padding=(ks - 1) // 2,
                              bias=True)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if not self.use_relu:
            x = self.relu(x)
        return x


class conv_block(nn.Module):
    # general conv block, 2 x (conv layers, batchnorm, relu)
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):
    # downsample by strided convolutions with stride 2, instead of maxpooling
    def __init__(self, ch_in, ch_out):
        super(down_conv).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=True), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class up_conv(nn.Module):
    # upsample convolutions in decoder
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    # ref: https://arxiv.org/abs/1804.03999
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Non_local_Attn(nn.Module):
    # non local attention block
    def __init__(self, channels):
        super(Non_local_Attn).__init__()
        self.in_channels = channels

        self.norm = Norm_layer(channels, type="group")
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)
        return x + A


class ChannelGate(nn.Module):
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

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                avg_pool = avg_pool.view(avg_pool.size(0), -1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                max_pool = avg_pool.view(avg_pool.size(0), -1)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(
            x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = single_conv(2, 1, kernel_size, use_relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class SpatialAttn(nn.Module):
    # spatial attention module
    # ref: https://arxiv.org/abs/1807.06521
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=["avg", "max"]):
        super(SpatialAttn, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio,
                                       pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class Encoder(nn.Module):
    # code ref: https://github.com/LeeJunHyun/Image_Segmentation/blob/db34de21767859e035aee143c59954fa0d94bbcd/network.py
    def __init__(self):
        super(Encoder).__init__()

        self.conv1 = conv_block(1, 64)
        self.down1 = down_conv(64, 64)  # 128->64
        self.conv2 = conv_block(64, 128)
        self.down2 = down_conv(128, 128)  # 64->32
        self.conv3 = conv_block(128, 256)
        self.down3 = down_conv(256, 256)  # 32->16
        self.conv4 = conv_block(256, 512)
        self.down4 = down_conv(512, 512)  # 16-> 8
        self.conv5 = conv_block(512, 1024)

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
        x = self.conv5(x)  # x should be b,1024,8,8 in latent space

        # return concat layers for connecting to decoder
        return x, (concat1, concat2, concat3, concat4)


class Decoder(nn.Module):
    def __init__(self):
        # axuiliary decoder for reconstruction task
        super(Decoder).__init__()

        # 8 ->16
        self.up1 = up_conv(1024, 512)
        self.upconv1 = conv_block(512, 512)
        # 16 -> 32
        self.up2 = up_conv(512, 256)
        self.upconv2 = conv_block(256, 256)
        # 32-> 64
        self.up3 = up_conv(256, 128)
        self.upconv3 = conv_block(128, 128)
        # 64 -> 128
        self.up4 = up_conv(64, 64)
        self.upconv4 = conv_block(64, 64)

        # 64 channel to 1
        self.final_conv = nn.Conv2d(64,
                                    1,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)

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

        x = F.tanh(
            x
        )  # map to [-1,1], depands on the range of input image, use sigmoid if input image is in [0,1]

        return x


# transformer bottleneck
class Transformer(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=1024,
                 nhead=12,
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

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        C = x.shape[1]
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # [B, H*W, C]
        x = x + self.pos_embedding
        return self.transformer(x)


# deocder - segmentation
class Segmentor(nn.Module):
    def __init__(self, num_cls):
        # segmentation head to produce segmentation map
        super(Decoder).__init__()

        # 8 ->16
        self.up5 = up_conv(1024, 512)
        self.upconv5 = conv_block(1024, 512)
        self.att5 = Attention_block(512, 512, 256)

        # 16 -> 32
        self.up4 = up_conv(512, 256)
        self.upconv4 = conv_block(512, 256)
        self.att4 = Attention_block(256, 256, 128)

        # 32-> 64
        self.up3 = up_conv(256, 128)
        self.upconv3 = conv_block(256, 128)
        self.att3 = Attention_block(128, 128, 64)

        # 64 -> 128
        self.up2 = up_conv(128, 64)
        self.upconv2 = conv_block(128, 64)
        self.att2 = Attention_block(64, 64, 32)

        # 64 channel to 1
        self.final_conv = nn.Conv2d(64,
                                    num_cls,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        self.seg_act = nn.Sigmoid()

    def forward(self, lat_feat, concat_feats):
        # channels: 64, 128, 256, 512
        concat1, concat2, concat3, concat4 = concat_feats

        x = self.up5(lat_feat)
        concat4 = self.att5(x, concat4)
        x = self.upconv5(torch.cat([concat4, x], dim=1))

        x = self.up4(x)
        concat3 = self.att4(x, concat3)
        x = self.upconv4(torch.cat([concat3, x], dim=1))

        x = self.up3(x)
        concat2 = self.att3(x, concat2)
        x = self.upconv3(torch.cat([concat2, x], dim=1))

        x = self.up2(x)
        concat1 = self.att2(x, concat1)
        x = self.upconv2(torch.cat([concat1, x], dim=1))

        x = self.final_conv(x)

        # segmentation map, sigmoid maps to [0,1]
        x = self.seg_act(x)  # TODO: check if sigmoid is needed

        return x


class MTUNet(nn.Module):
    # Multi-task Transformer U-Net

    def __init__(self, num_cls=1):
        super(MTUNet, self).__init__()
        self.encoder = Encoder()
        self.transformer = Transformer()
        self.decoder = Decoder()
        self.segmentor = Segmentor(num_cls)

    def forward(self, x):
        # x: [B, C, H, W]
        lat_feat, concat_feats = self.encoder(x)
        lat_feat = self.transformer(lat_feat)
        seg_map = self.segmentor(lat_feat, concat_feats)
        rec_img = self.decoder(lat_feat)
        return seg_map, rec_img