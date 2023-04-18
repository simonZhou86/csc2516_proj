import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from network import ATT_UNET
from network2 import MTUNet
from transBTS_2D import TransBTS
from baselines import UNet

from utils import AverageMeter
from metrics import dice, iou, filter_by_mask, _threshold

from dataset import BraTS_2d
from torchmetrics import Dice, JaccardIndex, Accuracy

def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def test_epoch(model, val_loader, device):

    dice_scores = AverageMeter()
    iou_scores = AverageMeter()
    acc_scores = AverageMeter()
    
    model.eval()
    for batch_idx, (img, target) in enumerate(val_loader):
        img, target = img.to(device), target.to(device)
        with torch.no_grad():
            pred_seg, pred_recon = model(img)
            pred_recon = F.tanh(pred_recon) #F.tanh(pred_recon)
            pred_seg = torch.sigmoid(pred_seg)
            
        
        # tseg_part_loss.update(seg_loss.item(), img.size(0))
        # taux_part_loss.update(aux_loss.item(), img.size(0))        
        #print(losses)
        dice_metric = Dice().to(device)
        #temp_pred_seg_bin = _threshold(pred_seg, 0.5)
        temp_ds = dice_metric(pred_seg, target.int())
        dice_scores.update(temp_ds.item(), img.size(0))
        #print(dice_scores)
        # TODO: add threshold value?
        #iou_metric = JaccardIndex(task = "binary", num_classes=2).to(device)
        temp_pred_seg_bin = _threshold(pred_seg, 0.5)
        acc_metric = Accuracy(task = 'binary').to(device)
        temp_acc = acc_metric(temp_pred_seg_bin, target.int())
        acc_scores.update(temp_acc.item(), img.size(0))
        
        filtered_pred, filtered_target = filter_by_mask(pred_seg, target.int())
        #filtered_pred_bin = _threshold(filtered_pred, 0.5)
        if filtered_pred.shape[0] == 0:
            temp_ious = 0
            iou_scores.update(0., 0)
        else:
            temp_ious = mIOU(torch.round(filtered_pred).int(), filtered_target.int())
            iou_scores.update(temp_ious.item(), filtered_pred.size(0))
    
    return dice_scores.avg, iou_scores.avg

def test_vis(model, device):
    test_img = torch.load("./data/test_img.pt").to(device)
    test_mask = torch.load("./data/test_mask.pt").to(device)
    # test_dataset = BraTS_2d("./data", mode='test', dev=False)
    
    # test_loader = DataLoader(test_dataset,
    #                         batch_size=4,
    #                         num_workers=1,
    #                         pin_memory=True,
    #                         shuffle=False)
    
    # dice_scores, iou_scores = test_epoch(model, test_loader, device)
    # print("Dice: {:.4f}, IOU: {:.4f}".format(dice_scores, iou_scores))

    selected = 2019
    #test_img_selected = test_img[selected].unsqueeze(0)
    test_img_selected = (2*test_img[selected]-1).unsqueeze(0) # should be 1,1,128,128

    pred_seg,_ = model(test_img_selected)
    pred_seg = torch.sigmoid(pred_seg)

    pred_seg = _threshold(pred_seg, 0.55)    
    return pred_seg.squeeze(0).squeeze(0), test_mask[selected].squeeze(0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTUNet(cross_att=True, recon=True).to(device)
    #model = ATT_UNET().to(device)
    #model = TransBTS().to(device)
    #model = UNet().to(device)
    
    my_model_para = torch.load("./checkpoints/mtunetCA_best_99.pth")
    #print(my_model_para.keys())
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in my_model_para.items():
        if ("num_batches_tracked" in k):
            continue
        else:
            name = k[7:] # remove 'module.' of data parallel
            new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    #print("finish!")
    pred, tar = test_vis(model, device)
    print(pred.shape, tar.shape)
    plt.figure(figsize=(5, 5))
    plt.subplot(1,2,1)
    plt.imshow(pred.detach().cpu().numpy(), cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(tar.detach().cpu().numpy(), cmap='gray')
    plt.show()
    