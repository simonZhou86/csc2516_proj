import torch
import wandb
import argparse
from torch.utils.data import DataLoader
from network import MTUNet
import time
from utils import AverageMeter
from metrics import dice, iou
import os
from loss import loss_func
from torchvision.models import vgg16_bn

vgg = vgg16_bn(pretrained=True)

def train_epoch(args, model, train_loader, optimizer, scheduler, device, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice_scores = AverageMeter()

    model.train()

    end = time.time()

    for batch_idx, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        pred_seg, pred_recon = model(img)
 
        loss = loss_func(vgg, pred_seg, pred_recon, target, 
                         args.c1, args.c2, 
                         args.lambda1, args.lambda2, 
                         args.block_idx, device)
        
        losses.update(loss.data[0], img.size(0))

        dice_scores.update(dice(pred_seg, target), img.size(0))

        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score {dice.val:.3f} ({dice.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, dice=dice_scores))
            
            wandb.log({"train_loss": losses.val,
                        "tran_data_time": data_time.val,
                        "train_batch_time": batch_time.val})
            
    wandb.log({"train_loss_epoch": losses.avg,
                "train_dice_epoch": dice_scores.avg,})

def test_epoch(args, model, val_loader, device, epoch):
    losses = AverageMeter()
    dice_scores = AverageMeter()

    model.eval()

    for batch_idx, (img, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)

        pred_seg, pred_recon = model(img)

        loss = loss_func(vgg, pred_seg, pred_recon, target, 
                         args.c1, args.c2, 
                         args.lambda1, args.lambda2, 
                         args.block_idx, device)
        
        losses.update(loss.data[0], img.size(0))

        dice_scores.update(dice(pred_seg, target), img.size(0))
            
        if batch_idx % args.log_interval == 0:
            print('Test: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Score {dice.val:.3f} ({dice.avg:.3f})'.format(
                    epoch, batch_idx, len(val_loader), loss=losses, dice=dice_scores))
            
    wandb.log({"val_loss_epoch": losses.avg,
                "val_dice_epoch": dice_scores.avg,})
        
    return dice_scores.avg

def train(args):
    train_dataset = None # TODO: create dataset
    val_dataset = None # TODO: create dataset
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MTUNet()
    model = torch.nn.DataParallel(model).to(device)
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(args.epochs):
        train_epoch(args, model, train_loader, optimizer, scheduler, device, epoch)
        test_epoch(args, model, val_loader, device, epoch)

        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)))

def test(args):
    test_dataset = None # TODO: create dataset
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=False)
    model = MTUNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.model_path))

    dice_score = test_epoch(args, model, test_loader, device, 0)
    print('Test result:\nDice score: {}'.format(dice_score))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train MTUNet')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--model_path', type=str, help='path to the model')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--recon_weight', type=float, default=1., help='weight of reconstruction loss')
    parser.add_argument('--log_interval', type=int, default=10, help='number of batches between logging')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save the model')
    parser.add_argument('--viz_wandb', type=str, default=None, help='wandb entity to log to')
    parser.add_argument('--c1', type=float, default=1., help='weight of segmentation loss')
    parser.add_argument('--c2', type=float, default=1., help='weight of reconstruction loss')
    parser.add_argument('--lambda1', type=float, default=1., help='weight of img_grad_dif loss')
    parser.add_argument('--lambda2', type=float, default=1., help='weight of percep loss')
    parser.add_argument('--block_idx', type=int, nargs='+', default=[0, 1, 2],
                         help='VGG block indices to use for style loss')
    args = parser.parse_args()
    if args.train:
        wandb.init(name="Train-MTUNet",
                   project="csc2516-project",
                   entity=args.viz_wandb)
        wandb.config = {
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "recon_weight": args.recon_weight,
        }
        train(args)
        wandb.finish()

    elif args.test:
        test(args)

