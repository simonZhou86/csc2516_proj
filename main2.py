# import torch
# import torch.nn.functional as F
# import wandb
# import argparse
# from torch.utils.data import DataLoader
# #from network2 import MTUNet, init_weights
# from transBTS_2D import TransBTS, init_weights
# import time
# from utils import AverageMeter
# from metrics import dice, iou, filter_by_mask, _threshold
# import os
# from loss2 import loss_func, loss_unet
# from torchvision.models import vgg16_bn
# from baselines import UNet
# from dataset import BraTS_2d
# from torchmetrics import Dice, JaccardIndex
# import torch.distributed as dist

# vgg = vgg16_bn(pretrained=True)

# def dice(output, target, eps=1e-5):
#     target = target.float()
#     num = 2 * (output * target).sum()
#     den = output.sum() + target.sum() + eps
#     return 1.0 - num/den

# def train_epoch(args, model, train_loader, optimizer, scheduler, device, epoch):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     seg_part_loss = AverageMeter()
#     aux_part_loss = AverageMeter()
#     dice_scores = AverageMeter() 
#     iou_scores = AverageMeter()

#     model.train()

#     end = time.time()

#     for batch_idx, (img, target) in enumerate(train_loader):
#         data_time.update(time.time() - end)

#         img, target = img.to(device), target.to(device)
#         img = img.float()
#         target = target.float() # cause error in BCE loss if target is long

#         optimizer.zero_grad()
#         if args.unet:
#             pred_seg = model(img)
#             pred_seg = torch.sigmoid(pred_seg)
#             loss = loss_unet(pred_seg, target, device)
#         else:
#             pred_seg, pred_recon = model(img)
#             pred_recon = torch.sigmoid(pred_recon) #F.tanh(pred_recon)
#             pred_seg = torch.sigmoid(pred_seg)
            
#             total_loss = dice(pred_seg, target)
            
#             # if isinstance(total_loss, tuple):
#             #     loss, seg_loss, aux_loss = total_loss
    
#             seg_loss = total_loss
        
#         losses.update(total_loss.item(), img.size(0))
#         seg_part_loss.update(seg_loss.item(), img.size(0))
#         #aux_part_loss.update(aux_loss.item(), img.size(0))
#         #print(losses)
#         dice_metric = Dice().to(device)
#         #temp_preg_seg_bin = _threshold(pred_seg, 0.5)
#         temp_ds = dice_metric(pred_seg, target.int())
#         dice_scores.update(temp_ds.item(), img.size(0))
#         #print(dice_scores)
#         # TODO: add threshold value?
#         iou_metric = JaccardIndex(task = "binary", num_classes=2).to(device)

#         filtered_pred, filtered_target = filter_by_mask(pred_seg, target.int())
#         #filtered_pred_bin = _threshold(filtered_pred, 0.5)
#         if filtered_pred.shape[0] == 0:
#             temp_ious = 0
#             iou_scores.update(0., 0)
#         else:
#             temp_ious = iou_metric(torch.round(filtered_pred), filtered_target)
#             iou_scores.update(temp_ious.item(), filtered_pred.size(0))
#         #print(iou_scores)

#         total_loss.backward()
#         optimizer.step()
#         scheduler.step()
#         batch_time.update(time.time() - end)
#         end = time.time()

#         # 'Aux Loss {loss_aux.val:.4f} ({loss_aux.avg:.4f})\t'
#         if batch_idx % args.log_interval == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Seg Loss {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
#                   'Dice Score {dice.val:.3f} ({dice.avg:.3f})'
#                   'IOU Score {iou.val:.3f} ({iou.avg:.3f})'.format(
#                     epoch, batch_idx, len(train_loader), batch_time=batch_time,
#                     data_time=data_time, loss=losses, loss_seg = seg_part_loss, dice=dice_scores, iou=iou_scores))
            
#             wandb.log({"train_loss": losses.val,
#                         "tran_data_time": data_time.val,
#                         "train_batch_time": batch_time.val})
            
#     wandb.log({"train_loss_epoch": losses.avg,
#                 "train_dice_epoch": dice_scores.avg,
#                 "train_iou_epoch": iou_scores.avg,})



# def test_epoch(args, model, val_loader, device, epoch):
#     losses = AverageMeter()
#     dice_scores = AverageMeter()
#     iou_scores = AverageMeter()
#     tseg_part_loss = AverageMeter()
#     taux_part_loss = AverageMeter()
    
#     val_recon_imgs = []
#     val_seg_maps = []
#     num_show = [int(i.numpy()) for i in torch.randperm(args.batch_size)[:min(5, args.batch_size)]]
#     model.eval()
#     for batch_idx, (img, target) in enumerate(val_loader):
#         img, target = img.to(device), target.to(device)

#         if args.unet:
#             pred_seg = model(img)
#             pred_seg = torch.sigmoid(pred_seg)
#             loss = loss_unet(pred_seg, target, device)
#         else:
#             pred_seg, pred_recon = model(img)
#             pred_recon = torch.sigmoid(pred_recon) #F.tanh(pred_recon)
#             pred_seg = torch.sigmoid(pred_seg)
#             # total_loss = loss_func(vgg, pred_seg, pred_recon, img, target, 
#             #                 args.c1, args.c2, 
#             #                 args.lambda1, args.lambda2, 
#             #                 args.block_idx, device, recon_loss=False)
            
#             total_loss = dice(pred_seg, target)
            
#             # if isinstance(total_loss, tuple):
#             #     loss, seg_loss, aux_loss = total_loss
#             #else:
        
#         losses.update(total_loss.item(), img.size(0))
#         tseg_part_loss.update(total_loss.item(), img.size(0))
#         #taux_part_loss.update(aux_loss.item(), img.size(0))        
#         #print(losses)
#         dice_metric = Dice().to(device)
#         #temp_pred_seg_bin = _threshold(pred_seg, 0.5)
#         temp_ds = dice_metric(pred_seg, target.int())
#         dice_scores.update(temp_ds.item(), img.size(0))
#         #print(dice_scores)
#         # TODO: add threshold value?
#         iou_metric = JaccardIndex(task = "binary", num_classes=2).to(device)
#         filtered_pred, filtered_target = filter_by_mask(pred_seg, target.int())
#         #filtered_pred_bin = _threshold(filtered_pred, 0.5)
#         if filtered_pred.shape[0] == 0:
#             temp_ious = 0
#             iou_scores.update(0., 0)
#         else:
#             temp_ious = iou_metric(torch.round(filtered_pred), filtered_target)
#             iou_scores.update(temp_ious.item(), filtered_pred.size(0))
        
#         # 'Aux Loss {loss_aux.val:.4f} ({loss_aux.avg:.4f})\t'
#         if batch_idx % args.log_interval == 0:
#             print('Test: [{0}][{1}/{2}]\t'
#                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                 'Seg Loss {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
#                 'Dice Score {dice.val:.3f} ({dice.avg:.3f})'
#                 'IOU Score {iou.val:.3f} ({iou.avg:.3f})'.format(
#                     epoch, batch_idx, len(val_loader), loss=losses, loss_seg = tseg_part_loss, dice=dice_scores, iou=iou_scores))

#     if not args.unet:
#         val_recon_imgs.extend([pred_recon[i].squeeze(0).detach().cpu().numpy() for i in num_show])
#     val_seg_maps.extend([pred_seg[i].squeeze(0).detach().cpu().numpy() for i in num_show])
    
#     if epoch % 10 == 0:
#         if not args.unet:
#             wandb.log({"val_loss_epoch": losses.avg,
#                         "val_dice_epoch": dice_scores.avg,
#                         "val_iou_epoch": iou_scores.avg,
#                         "axuilary recon images": [wandb.Image(i) for i in val_recon_imgs],
#                         "pred seg masks": [wandb.Image(i) for i in val_seg_maps]}, step = epoch)
#         else:
#             wandb.log({"val_loss_epoch": losses.avg,
#                         "val_dice_epoch": dice_scores.avg,
#                         "val_iou_epoch": iou_scores.avg,
#                         "pred seg masks": [wandb.Image(i) for i in val_seg_maps]})            
#     else:
#         wandb.log({"val_loss_epoch": losses.avg,
#                     "val_dice_epoch": dice_scores.avg,
#                     "val_iou_epoch": iou_scores.avg,}, step = epoch)
    

#     return dice_scores.avg, iou_scores.avg

# def train(args):
#     train_dataset = BraTS_2d(args.data_dir, mode='train', dev=args.dev)
#     val_dataset = BraTS_2d(args.data_dir, mode='val', dev=args.dev)
        
#     train_loader = DataLoader(train_dataset, 
#                               batch_size=args.batch_size, 
#                               num_workers=args.num_workers,
#                               pin_memory=True,
#                               shuffle=True)
    
#     val_loader = DataLoader(val_dataset, 
#                             batch_size=args.batch_size, 
#                             num_workers=args.num_workers,
#                             pin_memory=True,
#                             shuffle=False,
#                             drop_last=True)
    
#     if torch.cuda.is_available():
#         if args.slurm:
#             ngpus_per_node = torch.cuda.device_count()
#             local_rank = int(os.environ.get("SLURM_LOCALID"))
            
#             current_device = local_rank
#             torch.cuda.set_device(current_device)

#             devices = [
#                 torch.device(f"cuda:{i}") for i in range(ngpus_per_node)
#             ]
#             device = devices[current_device]

#             rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
#             print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
#             #init the process group
#             dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
#             print("process group ready!")
#         else:
#             device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu')
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     if args.unet:
#         model = UNet()
#         init_weights(model, init_type='kaiming')
#         model_name = 'unet'
        
#     else:
#         if args.cross_att:
#             model = MTUNet(cross_att=True)
#             init_weights(model, init_type='kaiming')
#             model_name = 'mtunet_CA'
#             print("running: ", model_name)
#         else:
#             model = TransBTS()#MTUNet(cross_att=True, recon=(not args.no_recon))
#             init_weights(model, init_type='kaiming')
#             model_name = 'baseline_tranBTS'#'baseline_transBTS'
#             print("running: ", model_name)

#     if args.slurm and torch.cuda.is_available():
#         print('From Rank: {}, ==> Making model..'.format(rank))
#         model.to(device)
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device], find_unused_parameters=True)
#     else:
#         model = torch.nn.DataParallel(model).to(device)
        
#     wandb.watch(model)

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)

#     for epoch in range(args.epochs):
#         train_epoch(args, model, train_loader, optimizer, scheduler, device, epoch)
#         test_epoch(args, model, val_loader, device, epoch)

#         torch.save(model.state_dict(), os.path.join(args.save_dir, f'{model_name}_{epoch}.pth'))

# def test(args):
#     test_dataset = BraTS_2d(args.data_dir, mode='test', dev=args.dev)
#     test_loader = DataLoader(test_dataset,
#                             batch_size=args.batch_size,
#                             num_workers=args.num_workers,
#                             pin_memory=True,
#                             shuffle=False)
#     model = TransBTS()#MTUNet()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = torch.nn.DataParallel(model).to(device)
#     model.load_state_dict(torch.load(args.model_path))

#     dice_score, iou_score = test_epoch(args, model, test_loader, device, 0)
#     print('Test result:\nDice score: {}\nIOU score: {}'.format(dice_score, iou_score))


# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description='Train MTUNet')
#     parser.add_argument('--train', action='store_true', help='train the model')
#     parser.add_argument('--test', action='store_true', help='test the model')
#     parser.add_argument('--data_dir', type=str, default='data', help='path to the dataset')
#     parser.add_argument('--model_path', type=str, default=None, help='path to load the model')
#     parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
#     parser.add_argument('--batch_size', type=int, default=8, help='batch size')
#     parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
#     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
#     parser.add_argument('--log_interval', type=int, default=10, help='number of batches between logging')
#     parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save the model')
#     parser.add_argument('--viz_wandb', type=str, default=None, help='wandb entity to log to')
#     parser.add_argument('--c1', type=float, default=1., help='weight of segmentation loss')
#     parser.add_argument('--c2', type=float, default=1., help='weight of reconstruction loss')
#     parser.add_argument('--lambda1', type=float, default=1., help='weight of img_grad_dif loss')
#     parser.add_argument('--lambda2', type=float, default=1., help='weight of percep loss')
#     parser.add_argument('--block_idx', type=int, nargs='+', default=[0, 1, 2],
#                          help='VGG block indices to use for style loss')
#     parser.add_argument('--cross_att', action='store_true', help='use cross attention in MTUNet?')
#     parser.add_argument('--unet', action='store_true', help='use UNet instead of MTUNet')
#     parser.add_argument('--dev', action='store_true', help='use dev mode')
#     parser.add_argument('--exp_name', type=str, default='Train-', help='experiment name')
#     parser.add_argument('--slurm', action='store_true',help='train on slurm server')
#     parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
#     parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
#     parser.add_argument('--no_recon', action='store_true', help='do not use reconstruction loss')
#     args = parser.parse_args()
    
#     print(args)
    
#     if args.dev:
#         args.epochs = 1
#         args.batch_size = 2
#         args.num_workers = 0
#         args.log_interval = 1

#     if args.train:
#         if args.unet:
#             wandb.init(name=args.exp_name + 'UNet',
#                    project="csc2516-localtest",
#                    entity=args.viz_wandb)
#         else:
#             wandb.init(name=args.exp_name + 'MTUNet',
#                     project="csc2516-localtest",
#                     entity=args.viz_wandb)
            
#         wandb.config = {
#             "max_epochs": args.epochs,
#             "batch_size": args.batch_size,
#             "lr": args.lr,
#         }
#         train(args)
#         wandb.finish()

#     elif args.test:
#         test(args)



import torch
import torch.nn.functional as F
import wandb
import argparse
from torch.utils.data import DataLoader
from network import MTUNet, init_weights, ATT_UNET
from baselines import UNet
from transBTS_2D import TransBTS 
import time
from utils import AverageMeter
from metrics import dice, iou, filter_by_mask, _threshold
import os
from loss import loss_func, loss_unet
from torchvision.models import vgg16_bn
from baselines import UNet
from dataset import BraTS_2d
from torchmetrics import Dice, JaccardIndex, Accuracy
import torch.distributed as dist

vgg = vgg16_bn(pretrained=True)

def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def train_epoch(args, model, train_loader, optimizer, scheduler, device, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_part_loss = AverageMeter()
    aux_part_loss = AverageMeter()
    dice_scores = AverageMeter() 
    iou_scores = AverageMeter()
    acc_scores = AverageMeter()

    model.train()

    end = time.time()

    for batch_idx, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img, target = img.to(device), target.to(device)
        img = img.float()
        target = target.float() # cause error in BCE loss if target is long

        optimizer.zero_grad()
        if args.unet:
            pred_seg = model(img)
            pred_seg = torch.sigmoid(pred_seg)
            loss = loss_unet(pred_seg, target, device)
        else:
            pred_seg, pred_recon = model(img)
            pred_recon = torch.sigmoid(pred_recon) #F.tanh(pred_recon)
            pred_seg = torch.sigmoid(pred_seg)
            
            total_loss = loss_func(vgg, pred_seg, pred_recon, img, target, 
                            args.c1, args.c2, 
                            args.lambda1, args.lambda2, 
                            args.block_idx, device, recon_loss=True)
            if isinstance(total_loss, tuple):
                loss, seg_loss, aux_loss = total_loss
            else:
                loss = total_loss
                seg_loss = total_loss
        
        losses.update(loss.item(), img.size(0))
        #seg_part_loss.update(seg_loss.item(), img.size(0))
        #aux_part_loss.update(aux_loss.item(), img.size(0))
        #print(losses)
        dice_metric = Dice().to(device)
        #temp_preg_seg_bin = _threshold(pred_seg, 0.5)
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
        #print(iou_scores)

        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        # 'Aux Loss {loss_aux.val:.4f} ({loss_aux.avg:.4f})\t'
                        #   'Seg Loss {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
                #   'Aux Loss {loss_aux.val:.4f} ({loss_aux.avg:.4f})\t'
        if batch_idx % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc Score {acc.val:.3f} ({acc.avg:.3f})'
                  'Dice Score {dice.val:.3f} ({dice.avg:.3f})'
                  'IOU Score {iou.val:.3f} ({iou.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc = acc_scores, dice=dice_scores, iou=iou_scores))
            
            wandb.log({"train_loss": losses.val,
                        "tran_data_time": data_time.val,
                        "train_batch_time": batch_time.val})
            
    wandb.log({"train_loss_epoch": losses.avg,
                "train_dice_epoch": dice_scores.avg,
                "train_iou_epoch": iou_scores.avg,})



def test_epoch(args, model, val_loader, device, epoch):
    losses = AverageMeter()
    dice_scores = AverageMeter()
    iou_scores = AverageMeter()
    tseg_part_loss = AverageMeter()
    taux_part_loss = AverageMeter()
    acc_scores = AverageMeter()
    
    val_recon_imgs = []
    val_seg_maps = []
    num_show = [int(i.numpy()) for i in torch.randperm(args.batch_size)[:min(5, args.batch_size)]]
    model.eval()
    for batch_idx, (img, target) in enumerate(val_loader):
        img, target = img.to(device), target.to(device)

        if args.unet:
            pred_seg = model(img)
            pred_seg = torch.sigmoid(pred_seg)
            loss = loss_unet(pred_seg, target, device)
        else:
            pred_seg, pred_recon = model(img)
            pred_recon = torch.sigmoid(pred_recon) #F.tanh(pred_recon)
            pred_seg = torch.sigmoid(pred_seg)
            total_loss = loss_func(vgg, pred_seg, pred_recon, img, target, 
                            args.c1, args.c2, 
                            args.lambda1, args.lambda2, 
                            args.block_idx, device, recon_loss=True)
            
            if isinstance(total_loss, tuple):
                loss, seg_loss, aux_loss = total_loss
            else:
                loss = total_loss
        
        losses.update(loss.item(), img.size(0))
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
        
        # 'Seg Loss {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t' 'Aux Loss {loss_aux.val:.4f} ({loss_aux.avg:.4f})\t'
        if batch_idx % args.log_interval == 0:
            print('Test: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc Score {acc.val:.3f} ({acc.avg:.3f})'
                'Dice Score {dice.val:.3f} ({dice.avg:.3f})'
                'IOU Score {iou.val:.3f} ({iou.avg:.3f})'.format(
                    epoch, batch_idx, len(val_loader), loss=losses, acc=acc_scores, dice=dice_scores, iou=iou_scores))

    if not args.unet:
        val_recon_imgs.extend([pred_recon[i].squeeze(0).detach().cpu().numpy() for i in num_show])
    val_seg_maps.extend([pred_seg[i].squeeze(0).detach().cpu().numpy() for i in num_show])
    
    if epoch % 10 == 0:
        if not args.unet:
            wandb.log({"val_loss_epoch": losses.avg,
                        "val_dice_epoch": dice_scores.avg,
                        "val_iou_epoch": iou_scores.avg,
                        "axuilary recon images": [wandb.Image(i) for i in val_recon_imgs],
                        "pred seg masks": [wandb.Image(i) for i in val_seg_maps]}, step = epoch)
        else:
            wandb.log({"val_loss_epoch": losses.avg,
                        "val_dice_epoch": dice_scores.avg,
                        "val_iou_epoch": iou_scores.avg,
                        "pred seg masks": [wandb.Image(i) for i in val_seg_maps]})            
    else:
        wandb.log({"val_loss_epoch": losses.avg,
                    "val_dice_epoch": dice_scores.avg,
                    "val_iou_epoch": iou_scores.avg,}, step = epoch)
    

    return dice_scores.avg, iou_scores.avg

def train(args):
    train_dataset = BraTS_2d(args.data_dir, mode='train', dev=args.dev)
    val_dataset = BraTS_2d(args.data_dir, mode='val', dev=args.dev)
        
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)
    
    if torch.cuda.is_available():
        if args.slurm:
            ngpus_per_node = torch.cuda.device_count()
            local_rank = int(os.environ.get("SLURM_LOCALID"))
            
            current_device = local_rank
            torch.cuda.set_device(current_device)

            devices = [
                torch.device(f"cuda:{i}") for i in range(ngpus_per_node)
            ]
            device = devices[current_device]

            rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
            print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
            #init the process group
            dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
            print("process group ready!")
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.unet:
        model = UNet()
        init_weights(model, init_type='kaiming')
        model_name = 'unet'
        
    else:
        if args.cross_att:
            model = MTUNet(cross_att=True)
            init_weights(model, init_type='kaiming')
            model_name = 'mtunetCA_best'
            print("running: ", model_name)
        else:
            model = MTUNet(cross_att=False, recon=(not args.no_recon))
            init_weights(model, init_type='kaiming')
            model_name = 'ours_w_attunet'#'baseline_transBTS'
            print("running: ", model_name)

    if args.slurm and torch.cuda.is_available():
        print('From Rank: {}, ==> Making model..'.format(rank))
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).to(device)
        
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(args.epochs):
        train_epoch(args, model, train_loader, optimizer, scheduler, device, epoch)
        test_epoch(args, model, val_loader, device, epoch)

        torch.save(model.state_dict(), os.path.join(args.save_dir, f'{model_name}_{epoch}.pth'))

def test(args):
    test_dataset = BraTS_2d(args.data_dir, mode='test', dev=args.dev)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=False)
    model = MTUNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.model_path))

    dice_score, iou_score = test_epoch(args, model, test_loader, device, 0)
    print('Test result:\nDice score: {}\nIOU score: {}'.format(dice_score, iou_score))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train MTUNet')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--data_dir', type=str, default='data', help='path to the dataset')
    parser.add_argument('--model_path', type=str, default=None, help='path to load the model')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--log_interval', type=int, default=10, help='number of batches between logging')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save the model')
    parser.add_argument('--viz_wandb', type=str, default=None, help='wandb entity to log to')
    parser.add_argument('--c1', type=float, default=1., help='weight of segmentation loss')
    parser.add_argument('--c2', type=float, default=1., help='weight of reconstruction loss')
    parser.add_argument('--lambda1', type=float, default=1., help='weight of img_grad_dif loss')
    parser.add_argument('--lambda2', type=float, default=1., help='weight of percep loss')
    parser.add_argument('--block_idx', type=int, nargs='+', default=[0, 1, 2],
                         help='VGG block indices to use for style loss')
    parser.add_argument('--cross_att', action='store_true', help='use cross attention in MTUNet?')
    parser.add_argument('--unet', action='store_true', help='use UNet instead of MTUNet')
    parser.add_argument('--dev', action='store_true', help='use dev mode')
    parser.add_argument('--exp_name', type=str, default='Train-', help='experiment name')
    parser.add_argument('--slurm', action='store_true',help='train on slurm server')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--no_recon', action='store_true', help='do not use reconstruction loss')
    args = parser.parse_args()
    
    print(args)
    
    if args.dev:
        args.epochs = 1
        args.batch_size = 2
        args.num_workers = 0
        args.log_interval = 1

    if args.train:
        if args.unet:
            wandb.init(name=args.exp_name + 'UNet',
                   project="csc2516-localtest",
                   entity=args.viz_wandb)
        else:
            wandb.init(name=args.exp_name + 'MTUNet',
                    project="csc2516-localtest",
                    entity=args.viz_wandb)
            
        wandb.config = {
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }
        train(args)
        wandb.finish()

    elif args.test:
        test(args)



