import time
import json
import random
import os, sys
import argparse
import numpy as np

import torch
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

from dataset.get_dataset import get_datasets
from models.EDL import EDLModel, AsymmetricBetaLoss
from utils.logger import setup_logger
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from utils.helper import function_mAP, get_raw_dict, ModelEma, add_weight_decay

np.set_printoptions(precision=4)

def parser_args():
    parser = argparse.ArgumentParser(description='Warmup Stage')

    # data
    parser.add_argument('--dataset_name', default='voc')
    parser.add_argument('--dataset_dir',  default='./data')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--output', default='./outputs')

    # train
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', 
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=300, type=int)
    parser.add_argument('--amp', action='store_true', default=True, help='apply amp')
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_epochs', default=12, type=int)
    parser.add_argument('--lb_ratio', default='0.1_1_1', type=str)

    parser.add_argument('--cutout', default=0.0, type=float,
                        help='cutout factor')

    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--net', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--is_data_parallel', action='store_true', default=False,
                        help='on/off nn.DataParallel()')
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')

    args = parser.parse_args()
    args.output = f'./outputs/{args.dataset_name}/{args.seed}/{args.lb_ratio}/warmup_abl_{args.warmup_epochs}'
    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, color=False, name="XXX")
    logger.info("Command: "+' '.join(sys.argv))
    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    return main_worker(args, logger)


def main_worker(args, logger):
    
    # load open-set data and update n_classes
    lb_train_dataset, ub_train_dataset, val_dataset = get_datasets(args, logger)

    # build model
    model = EDLModel(args.n_classes)
    if args.is_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay) # 0.9997

    lb_train_loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, 
        num_workers=args.workers, pin_memory=False
    )

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.warmup_epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: '
    )

    # optimizer
    optimizer = set_optimizer(model, args)
    args.steps_per_epoch = len(lb_train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        steps_per_epoch=args.steps_per_epoch, 
        epochs=args.warmup_epochs, 
        pct_start=0.2
    )

    end = time.time()
    best_ema_epoch = -1
    best_regular_epoch = -1
    best_mAP = 0
    best_ema_mAP = 0
    best_regular_mAP = 0
    ema_mAP_list = []
    regular_mAP_list = []
    if args.dataset_name.find('voc') != -1:
        criterion = AsymmetricBetaLoss(k=1, clip=0.2, gamma_neg=10)
    else:
        criterion = AsymmetricBetaLoss(k=1)

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.warmup_epochs):

        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(lb_train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluate on validation set
        mAP = validate(val_loader, model, args, logger)
        # mAP_ema = validate(val_loader, ema_m.module, args, logger)
        mAP_ema = -1

        mAPs.update(mAP)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.warmup_epochs - epoch - 1))

        regular_mAP_list.append(mAP)
        ema_mAP_list.append(mAP_ema)

        progress.display(epoch, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('val_mAP', mAP, epoch)
            summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

        # remember best (regular) mAP and corresponding epochs
        if mAP > best_regular_mAP:
            best_regular_mAP = mAP
            best_regular_epoch = epoch
        if mAP_ema > best_ema_mAP:
            best_ema_mAP = mAP_ema
            best_ema_epoch = epoch

        is_best = max(mAP, mAP_ema) > best_mAP
        if is_best:
            best_mAP = mAP

        logger.info(f'=> best ema mAP {best_ema_mAP} is in ep {best_ema_epoch}')
        logger.info(f'=> best regular mAP {best_regular_mAP} in ep {best_regular_epoch}')

        state_dict = model.state_dict()
        state_dict_ema = ema_m.module.state_dict()
        save_checkpoint({
            'epoch': epoch,
            'state_dict': state_dict,
            'state_dict_ema': state_dict_ema,
            'regular_mAP': regular_mAP_list,
            'ema_mAP': ema_mAP_list,
            'best_regular_mAP': best_regular_mAP,
            'best_ema_mAP': best_ema_mAP,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.output, 'warmup_model.pth.tar'))

    if summary_writer:
        summary_writer.close()
    
    return 0


def set_optimizer(model, args):
    if args.optim == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    elif args.optim == 'adamw':
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
        optimizer = getattr(torch.optim, 'AdamW')(
            param_dicts, args.lr,
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=args.weight_decay
        )
    return optimizer


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


def train(train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_per_epoch,
        [lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.warmup_epochs)
    )

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    model.train()
    for i, ((inputs_w, inputs_s), targets) in enumerate(train_loader):

        inputs = inputs_s.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=args.amp):
            alpha_c, beta_c, alpha_n, beta_n = model(inputs)

        l1 = criterion(alpha_c, beta_c, targets).sum()
        l2 = criterion(alpha_n, beta_n, targets).sum()
        loss = l1 + l2

        losses.update(loss.item(), inputs.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, args, logger):

    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(len(val_loader), [batch_time, mem], prefix='Test: ')

    model.eval()
    outputs_list1 = []
    outputs_list2 = []
    targets_list = []
    end = time.time()

    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            alpha_c, beta_c, alpha_n, beta_n = model(inputs)
            out1 = alpha_c / (alpha_c + beta_c)
            out2 = alpha_n / (alpha_n + beta_n)
        
        outputs_list1.append(out1.detach().cpu())
        outputs_list2.append(out2.detach().cpu())
        targets_list.append(targets.detach().cpu())

        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    labels = np.concatenate(targets_list)
    outputs1 = np.concatenate(outputs_list1)
    outputs2 = np.concatenate(outputs_list2)

    # calculate mAP
    mAP_c = function_mAP(labels, outputs1)
    mAP_n = function_mAP(labels, outputs2)
    logger.info("mAP_c: {}".format(mAP_c))
    logger.info("mAP_n: {}".format(mAP_n))
    return mAP_n


if __name__ == '__main__':
    main()
