import json
import time
import random
import os, sys
import argparse
import numpy as np

import torch
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

from models.EDL import EDLModel, AsymmetricBetaLoss, uncertainty
from dataset.get_dataset import get_datasets, TransformUnlabeled_WS, Mixed_mask_handler
from utils.logger import setup_logger
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from utils.helper import clean_state_dict, function_mAP, get_raw_dict, ModelEma, add_weight_decay


def parser_args():
    parser = argparse.ArgumentParser(description='Main')

    # data
    parser.add_argument('--dataset_name', default='coco')
    parser.add_argument('--dataset_dir',  default='./data')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--output', default='./outputs')

    # train
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,metavar='W', 
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=500, type=int)
    parser.add_argument('--amp', action='store_true', default=True, help='apply amp')
    parser.add_argument('--early_stop', default=3, type=int)
    parser.add_argument('--optim', default='adamw', type=str,
                        help='optimizer used')
    parser.add_argument('--warmup_epochs', default=12, type=int)
    parser.add_argument('--lb_ratio', default='0.1_1_1', type=str)

    parser.add_argument('--loss_k', default='2', type=int)
    parser.add_argument('--cutout', default=0.5, type=float, 
                        help='cutout factor')

    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    # model
    parser.add_argument('--net', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--is_data_parallel', action='store_true', default=False,
                        help='on/off nn.DataParallel()')
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--resume', default=None, type=str, 
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    if args.resume is None:
        args.resume = f'outputs/{args.dataset_name}/{args.seed}/{args.lb_ratio}/warmup_abl_12/warmup_model.pth.tar'
    args.output = f'outputs/{args.dataset_name}/{args.seed}/{args.lb_ratio}/abl_{args.loss_k}_{args.warmup_epochs}_{args.cutout}_{args.lr}_{args.batch_size}'

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

    lb_rate = len(lb_train_dataset) / (len(lb_train_dataset) + len(ub_train_dataset))
    args.lb_bs = max(int(args.batch_size * lb_rate), 4)
    args.ub_bs = args.batch_size - args.lb_bs

    # build model
    model = EDLModel(args.n_classes)
    if args.is_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()

    # load model
    if os.path.exists(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(os.path.join(args.resume))
        args.start_epoch = checkpoint['epoch'] + 1
        if 'state_dict' in checkpoint and 'state_dict_ema' in checkpoint:
            if args.dataset_name in ['nus']:
                state_dict = clean_state_dict(checkpoint['state_dict_ema'])
            else:
                state_dict = clean_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError("No model or state_dicr Found!!!")
        model.load_state_dict(state_dict, strict=False)
        logger.info(np.array(checkpoint['regular_mAP']))
        logger.info("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
        del checkpoint
        del state_dict
        torch.cuda.empty_cache() 
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))

    ema_m = ModelEma(model, args.ema_decay)

    lb_train_loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size,
        num_workers=args.workers, pin_memory=False)

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    optimizer = set_optimizer(model, args)

    end = time.time()
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    best_ema_epoch = -1
    regular_mAP_list = []
    ema_mAP_list = []

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):

        torch.cuda.empty_cache()

        pb_train_dataset = pseudo_label(ub_train_dataset, model, ema_m.module, args, logger)
        pb_train_loader = torch.utils.data.DataLoader(
            pb_train_dataset, batch_size=args.ub_bs,
            num_workers=args.workers, pin_memory=False
        )
        if epoch == args.warmup_epochs:
            lb_train_loader = torch.utils.data.DataLoader(
                lb_train_dataset, batch_size=args.lb_bs,
                num_workers=args.workers, pin_memory=False
            )
            optimizer = set_optimizer(model, args)
            args.steps_per_epoch = len(pb_train_loader)
            scheduler = lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr, 
                steps_per_epoch=args.steps_per_epoch, 
                epochs=args.epochs-args.warmup_epochs, pct_start=0.2
            )

        loss = semi_train(lb_train_loader, pb_train_loader, model, ema_m, optimizer, scheduler, epoch, args,logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluate on validation set
        mAP = validate(val_loader, model, args, logger)
        mAP_ema = validate(val_loader, ema_m.module, args, logger)

        mAPs.update(mAP)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.epochs - epoch - 1))

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

        logger.info(f'=> Best regular mAP {best_regular_mAP} in ep {best_regular_epoch}')
        logger.info(f'=> Best ema mAP {best_ema_mAP} in ep {best_ema_epoch}')

        # early stop
        if args.early_stop > 0 and epoch - best_ema_epoch >= args.early_stop:
            logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_ema_epoch))
            break

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
            param_dicts,args.lr,
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=args.weight_decay
        )
    return optimizer


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def pseudo_label(ub_train_dataset, model, ema_model, args, logger):

    model.eval()
    ema_model.eval()
    labels = []
    alphas_c = []
    betas_c = []
    alphas_n = []
    betas_n = []

    loader = torch.utils.data.DataLoader(
        ub_train_dataset, batch_size=args.test_batch_size,
        num_workers=args.workers, pin_memory=False
    )

    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=False)
    progress = ProgressMeter(len(loader), [batch_time, mem], prefix='Pseudo Labeling: ')
    end = time.time()

    for i, ((inputs_w, inputs_s), targets) in enumerate(loader):

        inputs_w = inputs_w.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            alpha_c, beta_c, alpha_n, beta_n = ema_model(inputs_w)

        alphas_c.append(alpha_c.detach().cpu().numpy())
        betas_c.append(beta_c.detach().cpu().numpy())
        alphas_n.append(alpha_n.detach().cpu().numpy())
        betas_n.append(beta_n.detach().cpu().numpy())
        labels.append(targets.numpy())

        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    alphas_c = np.concatenate(alphas_c)
    betas_c = np.concatenate(betas_c)
    alphas_n = np.concatenate(alphas_n)
    betas_n = np.concatenate(betas_n)
    labels = np.concatenate(labels)

    u_c = uncertainty(alphas_c, betas_c, args.n_classes)
    # u_n = uncertainty(alphas_n, betas_n, args.n_classes)

    ub_train_imgs = ub_train_dataset.X

    u = normalization(u_c)
    mask = np.expand_dims(1-u, 1).repeat(args.n_classes, 1)

    outputs1 = alphas_c / (alphas_c + betas_c)
    outputs2 = alphas_n / (alphas_n + betas_n)
    pseudo_labels = outputs1 * outputs2
    
    pb_train_dataset = Mixed_mask_handler(
        ub_train_imgs, pseudo_labels, mask, transform=TransformUnlabeled_WS(args))
    pb_mAP = function_mAP(labels, pseudo_labels)
    logger.info(f'pb_mAP {pb_mAP}')
    pred1_mAP = function_mAP(labels, outputs1)
    logger.info(f'pred1_mAP {pred1_mAP}')
    pred2_mAP = function_mAP(labels, outputs2)
    logger.info(f'pred2_mAP {pred2_mAP}')

    return pb_train_dataset


def semi_train(lb_train_loader, ub_train_loader, model, ema_m, 
               optimizer, scheduler, epoch, args, logger):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    loss_lb = AverageMeter('L_lb', ':5.3f')
    loss_ub = AverageMeter('L_ub', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(ub_train_loader),
        [loss_lb, loss_ub, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs)
    )

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    loss_func_lb = AsymmetricBetaLoss(clip=0.2)
    loss_func_ub = AsymmetricBetaLoss(clip=0.05, k=args.loss_k)

    # switch to train mode
    model.train()
    lb_train_iter = iter(lb_train_loader)
    for i, ((inputs_w_ub, inputs_s_ub), labels_ub, mask) in enumerate(ub_train_loader):

        try:
            (_, inputs_s_lb), labels_lb = next(lb_train_iter)
        except:
            lb_train_iter = iter(lb_train_loader)
            (_, inputs_s_lb), labels_lb = next(lb_train_iter)

        n_lb = labels_lb.shape[0]
        n_ub = labels_ub.shape[0]
        inputs = torch.cat([inputs_s_lb, inputs_s_ub], dim=0).cuda(non_blocking=True)
        labels = torch.cat([labels_lb, labels_ub], dim=0).cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            alpha_c, beta_c, alpha_n, beta_n = model(inputs)


        alpha_c_lb, alpha_c_ub = alpha_c[:n_lb], alpha_c[n_lb:]
        beta_c_lb, beta_c_ub = beta_c[:n_lb], beta_c[n_lb:]
        alpha_n_lb, alpha_n_ub = alpha_n[:n_lb], alpha_n[n_lb:]
        beta_n_lb, beta_n_ub = beta_n[:n_lb], beta_n[n_lb:]
        labels_lb, labels_ub = labels[:n_lb], labels[n_lb:]

        L_lb = loss_func_lb(alpha_c_lb, beta_c_lb, labels_lb).sum()
        L_lb += loss_func_lb(alpha_n_lb, beta_n_lb, labels_lb).sum()
        L_ub = loss_func_ub(alpha_n_ub, beta_n_ub, labels_ub)
        L_ub = (L_ub * mask).sum()
        loss = L_lb + L_ub

        # record loss
        loss_lb.update(L_lb.item(), inputs_s_lb.size(0))
        loss_ub.update(L_ub.item(), inputs_s_ub.size(0))
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
