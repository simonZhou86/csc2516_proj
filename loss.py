# Loss functions for the project

# Author: Simon, last modify Nov. 18, 2022

"""
Change log:
- Reacher: file created, implement L1 loss and L2 loss function
- Reacher: update image gradient calculation
- Simon: update image gradient loss
- Simon: add loss_func2, and L1_Charbonnier_loss
"""

import numpy as np
import torch
import torch.nn as nn
from our_utils import Percep_loss
from torchmetrics.functional import image_gradients
from torchvision.transforms import transforms
import torch.nn.functional as F

class PercepHook:
    '''
    Pytorch forward hook for computing the perceptual loss
    without modifying the original VGG16 network
    '''
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


class Percep_loss(nn.Module):
    '''
    compute perceptual loss between fused image and input image
    or we can use LIPIS implementation
    '''
    def __init__(self, vgg, block_idx, device):
        '''
        block_index: the index of the block in VGG16 network, int or list
        int represents single layer perceptual loss
        list represents multiple layers perceptual loss
        '''
        super(Percep_loss, self).__init__()
        self.block_idx = block_idx
        self.device = device
        # load vgg16_bn model features
        self.vgg = vgg.features.to(device).eval()
        #self.loss = nn.MSELoss()

        # unable gradient update
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # remove maxpooling layer and relu layer
        # TODO:check this part on whether we want relu or not
        bns = [i - 2 for i, m in enumerate(self.vgg) if isinstance(m, nn.MaxPool2d)]

        # register forward hook
        self.hooks = [PercepHook(self.vgg[bns[i]]) for i in block_idx]
        self.features = self.vgg[0: bns[block_idx[-1]] + 1]

    def forward(self, inputs, targets):
        '''
        compute perceptual loss between inputs and targets
        '''
        if inputs.shape[1] == 1:
            # expand 1 channel image to 3 channel, [B, 1, H, W] -> [B, 3, H, W]
            inputs = inputs.expand(-1, 3, -1, -1)
        if targets.shape[1] == 1:
            targets = targets.expand(-1, 3, -1, -1)
        
        # get vgg output
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        assert len(input_features) == len(target_features), 'number of input features and target features should be the same'
        loss = 0
        for i in range(len(input_features)):
            #loss += self.loss(input_features[i], target_features[i]) # mse loss
            loss += ((input_features[i] - target_features[i]) ** 2).mean() # l2 norm
        
        return loss


class grad_loss(nn.Module):
    '''
    image gradient loss
    '''
    def __init__(self, device, amp = True, vis = False):

        super(grad_loss, self).__init__()
        
        
            #with torch.cuda.amp.autocast(enabled=amp):
        kernel_x = torch.Tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = torch.Tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])

        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        # do not want update these weights
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).to(device)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).to(device)
        #self.weight_yx = nn.Parameter(data=kernel_x.double(), requires_grad=False).to(device)
        #self.weight_yy = nn.Parameter(data=kernel_y.double(), requires_grad=False).to(device)
        self.vis = vis
    
    def forward(self, x, y):
        # conv2d to find image gradient in x direction and y direction
        # of input image x and image y
        grad_xx = F.conv2d(x, self.weight_x)
        grad_xy = F.conv2d(x, self.weight_y)
        grad_yx = F.conv2d(y, self.weight_x)
        grad_yy = F.conv2d(y, self.weight_y)

        if self.vis:
            return grad_xx, grad_xy, grad_yx, grad_yy
        
        # total image gradient, in dx and dy direction for image X and Y
        # gradientX = torch.abs(grad_xx) + torch.abs(grad_xy)
        # gradientY = torch.abs(grad_yx) + torch.abs(grad_yy)
        x_diff = ((torch.abs(grad_xx) - torch.abs(grad_yx)) ** 2).mean()
        y_diff = ((torch.abs(grad_xy) - torch.abs(grad_yy)) ** 2).mean()
        
        # mean squared frobenius norm (||.||_F^2)
        #grad_f_loss = torch.mean(torch.pow(torch.norm((gradientX - gradientY), p = "fro"), 2))
        grad_f_loss = x_diff + y_diff
        return grad_f_loss


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, x, y):
        # x: predict, y: target
        loss = torch.mean(torch.sqrt((x - y)**2 + self.eps))
        return loss


def l1_loss(predicted, target):
    """
    To compute L1 loss using predicted and target
    """
    return torch.abs(predicted - target).mean()


def mse_loss(predicted, target):
    """
    To compute L2 loss between predicted and target
    """
    return torch.pow((predicted - target), 2).mean()
    #return torch.mean(torch.pow(torch.norm((predicted - target), p = "fro"), 2))





def perceptual_loss(vgg, predicted, target, block_idx, device):
    """
    compute perceptual loss between predicted and target
    """
    p_loss = Percep_loss(vgg, block_idx, device)
    return p_loss(predicted, target)


def loss_func(vgg, predicted, target, lambda1, lambda2, block_idx, device):
    """
    same as loss_func, except the gradient loss is change to grad_loss() class
    """
    img_grad_loss = grad_loss(device)
    #L1_charbonnier = L1_Charbonnier_loss()
    #reg_loss = L1_charbonnier(predicted, target)
    reg_loss = mse_loss(predicted, target)
    img_grad_dif = img_grad_loss(predicted, target)
    percep = perceptual_loss(vgg, predicted, target, block_idx, device)
    loss = reg_loss + lambda1 * img_grad_dif + lambda2 * percep
    return loss, reg_loss, img_grad_dif, percep
