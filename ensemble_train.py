import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# import models
import ensemble_model as models
import numpy as np
import matplotlib.pyplot as plt

from data import RoadAllData, build_transform, build_special_transform
from loss import *
from sklearn.metrics import roc_auc_score
import warnings
import cv2


parser = argparse.ArgumentParser(description='PyTorch Lane-line Anomaly Detection Training')

# 数据、模型保存、模型checkpoint加载的路径
parser.add_argument('--data', metavar='DIR', default='/data2/zrliu/dataset_cmp/dataset/', help='path to dataset')
parser.add_argument('--save_path', type=str, default='/data2/zrliu/checkpoints/hwcmp/', help='path to save model checkpoint.')
parser.add_argument('--resume', nargs='+', default=[''], type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--val', type=str, default='val', help='val dataset name.')

# 网络训练相关超参数

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--num_classes', type=int, default=8, help='number of classes.')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by a ratio)')                    
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--warmup', default=0, type=int, help='epochs to warm-up')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--labelsmooth', type=float, default=0.0, help='label smooth epsilon of loss function')
parser.add_argument('--alpha', default=[0.5, 0.5], nargs=2, type=float, help='class balance weight of loss function')
parser.add_argument('--gamma', type=float, default=0.0, help='focal loss factor of loss function')
parser.add_argument('--mask_rate', type=float, default=[0.0, 0.0], nargs=2, help='mask rate of dominant categories')
parser.add_argument('--weight', action='store_true', help='whether to use weighted cross entropy depending on numbers of each category')
parser.add_argument('--tri', type=float, default=0.0, help='use triplet loss')
parser.add_argument('--logtri', type=float, default=0.0, help='weight of logictriplet loss')
parser.add_argument('--multi_cls', action='store_true', help='use multi-classification head')
parser.add_argument('--multi_pos', action='store_true', help='use multi-positive loss')
parser.add_argument('--multi_scale', action='store_true', help='use multi-scale feature fusion')
parser.add_argument('--bce', action='store_true', help='whether to use binary cross-entropy loss when using multi-classifier')

# 测试专用超参数，如果置为True，则只计算测试集结果，需要同时使用resume参数加载模型checkpoint
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# DDP 分布式训练超参数(基本不用管, 用默认值就OK)
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

# 关于我们的方法的超参数
parser.add_argument('--unlabeled', action='store_true', help='use unlabeled data.')
parser.add_argument('--conloss', action='store_true', help='use constrast_loss.')
parser.add_argument('--split', type=str, default='37', help='split of dataset(37 or 46).')

parser.add_argument('--expand', action='store_true', help='whether to expand train dataset to balance samples among different classes.')
parser.add_argument('--scale_length', type=float, default=1, help="scale torch dataset's length.")

parser.add_argument('--img_size', nargs=2, type=int, default=[2400,1080], help='image size')
parser.add_argument('--crop_rate', type=float, default=1.142, help='rate to enlarge image before rand crop (for default, rand crop is disbaled).')
parser.add_argument('--flip', type=str, nargs='+', default=['h'], help='whether to use rand flip("h" : horizon flip | "v" : vertical flip).')
parser.add_argument('--special_transform', action='store_true', help='whether to use special transform.')
parser.add_argument('--special_prob', type=float, default=0.3, help='probability of performing a special transformation on an image.')
parser.add_argument('--updown_crop', type=float, nargs='+', default=[0, 0, 0], help='Probability of top, undertop, bottom crop')
parser.add_argument('--translate', action='store_true', help='whether to use random translate')
parser.add_argument('--rotate_prob', type=float, default=0, help='probability to use rand rotate.')
parser.add_argument('--canny', action='store_true', help='whether to use canny.')
parser.add_argument('--hist', action='store_true', help='whether to use hist mode.')
parser.add_argument('--gasuss_noise', action='store_true', help='gasuss_noise augmentation.')
parser.add_argument('--hist_level', type=int, default=16, help='hist level.')
parser.add_argument('--old_version', action='store_true', help='to support old checkpoint.')
parser.add_argument('--show', action='store_true', help='show images after transforms.')

parser.add_argument('--colorjittor', type=float, nargs=4, default=[0, 0, 0, 0], help='color jittor.')
parser.add_argument('--mixup', action='store_true', help='mixup augmentation.')
parser.add_argument('--mixup_alpha', type=float, default=1., help='mixup alpha value.')
parser.add_argument('--moco_pretrained', nargs='+', type=str, default=[''], help='checkpoint path of moco pretrained model.')

parser.add_argument('--deepfuse', action='store_true', help='whether to use deepfuse model.')
parser.add_argument('-a', '--archs', nargs='+', metavar='ARCH', default=['resnet18'], help='model architecture: ' +' | '+' (default: resnet18)')

best_auc = 0

def count_parameters(model): # 用来计算模型参数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_auc
    args.gpu = gpu
    
    assert len(args.moco_pretrained) == len(args.resume)
    num_models = len(args.resume)
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    args.arch = 'EnsembleDeepFuseModel' if args.deepfuse else 'EnsembleModel'
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](args.archs, pretrained=True, auxiliary=args.conloss, num_classes=args.num_classes, multi_cls=args.multi_cls, multi_scale=args.multi_scale)
    else:  
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](args.archs, auxiliary=args.conloss, num_classes=args.num_classes, multi_cls=args.multi_cls, multi_scale=args.multi_scale)

    print("model has %.2fM trainable parameters in total." % (count_parameters(model) / 1e6))

    for i in range(num_models):
        if args.moco_pretrained[i] not in ['', 'None', 'none']:
            checkpoint = torch.load(args.moco_pretrained[i])
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.classifier[i].load_state_dict(state_dict, strict=False)
            flag = True
            for m in msg.missing_keys:
                if not m.startswith('fc'):  
                    flag = False
                    break
            assert flag, 'load moco checkpoint failed!'
            print('successfully loaded moco checkpoint!')     
    
    # optionally resume from a checkpoint
    for i in range(num_models):
        if args.resume[i] not in ['', 'None', 'none']:
            if os.path.isfile(args.resume[i]):
                print("=> loading checkpoint '{}'".format(args.resume[i]))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume[i])
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume[i], map_location=loc)           
            
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.'):
                        # remove prefix
                        state_dict[k[len("module."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                model.classifiers[i].load_state_dict(checkpoint['state_dict'], strict=True)
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
                
                if args.evaluate and 'args' in checkpoint:
                    print('\n''\n')
                    print("config of checkpoint: ")
                    print('--------------------------------------------------------------------------------------------------')
                    # Print arguments
                    for k in checkpoint['args'].__dict__:
                        print("{:40}{}".format(k, str(checkpoint['args'].__dict__[k])))
                    print('--------------------------------------------------------------------------------------------------')
                
                                            
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.unlabeled:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if args.unlabeled:
                model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.conloss:
        criterion = maskedMultiCEloss(args.num_classes, weight=args.weight).cuda(args.gpu)
    else:
        criterion = {}
        criterion['ce'] = multiCEloss(args, args.num_classes, args.labelsmooth, args.alpha, args.gamma).cuda(args.gpu)
        criterion['tri'] = OriTripletLoss(margin=0.5).cuda(args.gpu)
        criterion['logtri'] = LogicTripletLoss(margin=0).cuda(args.gpu)

    if 'swin' in args.arch:
        print('Optimizier: AdamW')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    transform_train = build_transform(size=(args.img_size[0], args.img_size[1]), rate_for_crop=args.crop_rate, rand_filp=args.flip, to='Tensor', norm=True, 
        updown_rand_crop=args.updown_crop, gasuss_noise=args.gasuss_noise, translate=args.translate, canny=args.canny, hist=args.hist, hist_level=args.hist_level, colorjittor=args.colorjittor)
    transform_val = build_transform(size=(args.img_size[0], args.img_size[1]), rate_for_crop=None, rand_filp=False, to='Tensor', norm=True, canny=args.canny, hist=args.hist, hist_level=args.hist_level)
    

    if args.special_transform:
        special_transform = build_special_transform()
    else:
        special_transform = None

    train_dataset = RoadAllData(args.data, transform=transform_train, 
                                sepcial_transform=special_transform,
                                sepcial_prob=args.special_prob,
                                two_transforms=args.conloss,
                                load_unlabeled=args.unlabeled, 
                                expand=args.expand, 
                                scale_length=args.scale_length, split=args.split
    )
    
    
    val_dataset = RoadAllData(args.data, transform=transform_val, mode=args.val, expand=False, split=args.split)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    else:
        train_sampler = None



    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.conloss:
            train_one_epoch_with_conloss(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node)
        else:
            train_one_epoch_without_conloss(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node)
        
        # evaluate on validation set
        acc1, auc = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = (auc > best_auc)
        best_auc = max(auc, best_auc)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'args' : args,
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_path, 'checkpoint_{}.pth.tar'.format(epoch))


def train_one_epoch_with_conloss(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node):
    contra_criterion = SupConLoss(device='cuda:{}'.format(args.gpu))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.rotate_prob > np.random.rand():
            images[0] = images[0].permute(0, 1, 3, 2)
        if args.show and (args.rank % ngpus_per_node == 0):
            img1 = images[0][0].permute(1,2,0).detach().cpu().numpy()
            img2 = images[1][0].permute(1,2,0).detach().cpu().numpy()
            min_ = [min(img1[:, :, j].min(), img2[:, :, j].min()) for j in range(3)]
            max_ = [max(img1[:, :, j].max(), img2[:, :, j].max()) for j in range(3)]
            img1[:, :, 0] = (img1[:, :, 0] - min_[0]) / (max_[0] - min_[0])
            img1[:, :, 1] = (img1[:, :, 1] - min_[1]) / (max_[1] - min_[1])
            img1[:, :, 2] = (img1[:, :, 2] - min_[2]) / (max_[2] - min_[2])
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            dt = (max(img1.shape[0], img2.shape[0]) - img1.shape[0]) // 2
            db = max(img1.shape[0], img2.shape[0]) - img1.shape[0] - dt
            dl = (max(img1.shape[1], img2.shape[1]) - img1.shape[1]) // 2
            dr = max(img1.shape[1], img2.shape[1]) - img1.shape[1] - dl
            img1 = cv2.copyMakeBorder(img1, dt, db, dl, dr, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            img2[:, :, 0] = (img2[:, :, 0] - min_[0]) / (max_[0] - min_[0])
            img2[:, :, 1] = (img2[:, :, 1] - min_[1]) / (max_[1] - min_[1])
            img2[:, :, 2] = (img2[:, :, 2] - min_[2]) / (max_[2] - min_[2])
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            img = np.concatenate([img1, np.ones((img1.shape[0], int(img1.shape[1] * 0.1), 3)), img2], axis=1)
            cv2.imshow('img', img)
            key = cv2.waitKey()
            if key == ord('q'): 
                args.show = False
                cv2.destroyAllWindows()

        if args.gpu is not None:  
            # images = torch.cat([images[0], images[1]], dim=0)
            images[0] = images[0].cuda(args.gpu)
            images[1] = images[1].cuda(args.gpu)
            # images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        
        noise = torch.rand(args.num_classes+1).cuda(args.gpu)
        ids_shuffle = torch.argsort(noise)
        contra_target = target[:, ids_shuffle].argmax(dim=1)
        contra_target = ids_shuffle[contra_target]
        contra_target[contra_target == args.num_classes] = target[contra_target == args.num_classes, args.num_classes]

        # compute output
        outs0, feat_list0 = model(images[0])
        outs1, feat_list1 = model(images[1])
        outs, feat_list = [], []
        for out0, out1 in zip(outs0, outs1):
            outs.append(torch.cat([out0, out1], dim=0))
        for feat0, feat1 in zip(feat_list0, feat_list1):
            feat_list.append(torch.cat([feat0, feat1], dim=0))
        
        
        bsz = target.size(0)
        for j in range(len(outs)):  outs[j] = outs[j][:bsz] # 一张图像的两次数据增广 只取其中一次来做 CE loss
        loss = 0
        mask = (target[:, args.num_classes] == 0)
        if mask.sum():
            loss += criterion(outs, target, mask)
        
        for index in range(len(feat_list)):
            features = feat_list[index]
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss += contra_criterion(features, labels=contra_target) * 1e-1

        # measure accuracy and record loss
        losses.update(loss.item(), images[0].size(0))
        if mask.sum():
            acc1, = accuracy(outs[0][mask, :], target[mask, 0], topk=(1,))
            top1.update(acc1[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i) 

def train_one_epoch_without_conloss(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ce = AverageMeter('Loss_ce', ':.4e')
    losses_tri = AverageMeter('Loss_tri', ':.4e')
    # losses_logtri = AverageMeter('Loss_logtri', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
    len(train_loader),
    [batch_time, data_time, losses_ce, losses_tri, top1],
    prefix="Epoch: [{}]".format(epoch))
    ce_criterion = criterion['ce']
    tri_criterion = criterion['tri']
    # logtri_criterion = criterion['logtri']

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.rotate_prob > np.random.rand():
            images = images.permute(0, 1, 3, 2)
        
        if args.gpu is not None:  
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        
        b = images.shape[0]
        if args.mixup:
            clone_imgs = images.clone()
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            index = torch.randperm(b).cuda()
            mix_imgs = lam * clone_imgs + (1 - lam) * clone_imgs[index, :]
            images = torch.cat([images, mix_imgs], dim=0)
        
        if args.show and (args.rank % ngpus_per_node == 0):
            img = images[-1].permute(1,2,0).detach().cpu().numpy().copy()
            # img = img * 0.5 + 0.5
            img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].min()) / (img[:, :, 0].max() - img[:, :, 0].min())
            img[:, :, 1] = (img[:, :, 1] - img[:, :, 1].min()) / (img[:, :, 1].max() - img[:, :, 1].min())
            img[:, :, 2] = (img[:, :, 2] - img[:, :, 2].min()) / (img[:, :, 2].max() - img[:, :, 2].min())
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imshow('img', img)
            # key = cv2.waitKey()
            plt.imshow(img)
            plt.show()
            # if key == ord('q'): 
            #     args.show = False
            #     cv2.destroyAllWindows()



        outs, feats = model(images)
        loss_ce = ce_criterion([out[:b, :] for out in outs], target)
        loss_tri, _ = tri_criterion(feats[:b, :], target[:,0])
        # loss_logtri, _ = logtri_criterion(outs[0][:b, :], target[:,0])
        if args.mixup:  loss_ce = loss_ce + lam * ce_criterion([out[b:, :] for out in outs], target) + \
                                (1 - lam) * ce_criterion([out[b:, :] for out in outs], target[index])
        loss = loss_ce + args.tri * loss_tri
        # measure accuracy and record loss
        losses_ce.update(loss_ce.item(), b)
        losses_tri.update(loss_tri.item(), b)
        # losses_logtri.update(loss_logtri.item(), b)

        acc1, = accuracy(outs[0][:b, :], target[:, 0], topk=(1,))
        top1.update(acc1[0], b)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_ce = AverageMeter('Loss', ':.4e')
    losses_tri = AverageMeter('Loss', ':.4e')
    # losses_logtri = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_ce, losses_tri, top1],
        prefix='Test: ')

    num_classes = 2 if args.multi_cls else args.num_classes 
    ce_criterion_ = multiCEloss(args, args.num_classes, args.labelsmooth, args.alpha, args.gamma).cuda(args.gpu)
    tri_criterion_ = OriTripletLoss(margin=0.5)
    # logtri_criterion_ = LogicTripletLoss(margin=0)
    # switch to evaluate mode
    model.eval()
    total_pred = []
    total_target = []
    res_matrix = torch.zeros((2, 8)).cuda(args.gpu)
    cls_matrixs = [torch.zeros((2, args.num_classes)).cuda(args.gpu) for _ in range(num_classes - 1)]
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            outs, feats = model(images)
            loss_ce = ce_criterion_(outs, target)
            loss_tri, _ = tri_criterion_(feats, target[:,0])
            # loss_logtri, _ = logtri_criterion_(outs[0], target[:, 0])
            preds = outs[0].argmax(dim=1).detach().cpu().numpy()
            
            res_matrix[0, :] += target[preds == 1, :8].sum(dim=0)
            res_matrix[1, :] += target[preds == 0, :8].sum(dim=0)

            for cls_i in range(1, num_classes):
                preds_cls = outs[cls_i].argmax(dim=1).detach().cpu().numpy()
                cls_matrixs[cls_i - 1][0, :] += target[preds_cls != 0, :args.num_classes].sum(dim=0)
                cls_matrixs[cls_i - 1][1, :] += target[preds_cls == 0, :args.num_classes].sum(dim=0)
            
            # measure accuracy and record loss
            # outs[1] = torch.stack([1.0 - outs[1][:,0], outs[1][:,0]], dim=1)
            total_pred.extend(list(outs[0].softmax(dim=1)[:, 0].detach().cpu().numpy()))
            total_target.extend(list((1-target[:, 0]).detach().cpu().numpy()))
            acc1, = accuracy(outs[0], target[:, 0], topk=(1,))
            losses_ce.update(loss_ce.item(), images.size(0))
            losses_tri.update(loss_tri.item(), images.size(0))
            # losses_logtri.update(loss_logtri.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        total_pred = np.array(total_pred)
        total_target = np.array(total_target)
        auc = get_auc(total_pred, total_target)
        # TODO: this should also be done with the ProgressMeter

        if args.rank == 0:
            print(' * Acc@1 {top1.avg:.3f}, * AUC {auc:.3f}'.format(top1=top1, auc=auc))
            res_matrix /= res_matrix.sum(dim=0, keepdims=True)
            res_matrix = res_matrix.detach().cpu().numpy()
            print('classifier 0', end='\t\t')
            for i in range(8):  print(i, end='\t\t')
            print()
            for i in range(2):
                print(i, end='\t\t')
                for j in range(8):
                    print(round(res_matrix[i, j], 2), end='\t\t')
                print()

            for cls_i in range(0, num_classes - 1) :
                cls_matrixs[cls_i] /= cls_matrixs[cls_i].sum(dim=0, keepdims=True)
                cls_matrixs[cls_i] = cls_matrixs[cls_i].detach().cpu().numpy()
                print('classifier ' + str(cls_i + 1), end='\t\t')
                for i in range(8):  print(i, end='\t\t')
                print()
                for i in range(2):
                    print(i, end='\t\t')
                    for j in range(args.num_classes):
                        print(round(cls_matrixs[cls_i][i, j], 2), end='\t\t')
                    print()
    return top1.avg, auc


def save_checkpoint(state, is_best, basepath, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(basepath, filename))
    if is_best:
        shutil.copyfile(os.path.join(basepath, filename), os.path.join(basepath, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




def get_auc(y_pred, y_true):
    return roc_auc_score(y_true, y_pred, average='macro')


# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     for milestone in args.schedule:
#         lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch < args.warmup:
        lr = args.lr * (epoch * (100.0 / args.warmup) + 1) / 100.0
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()