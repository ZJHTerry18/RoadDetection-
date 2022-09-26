import torch
import models
from data import build_transform, RoadAllData, RoadAllDataTest
from sklearn.metrics import roc_auc_score
import argparse
import time
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
import pandas as pd
import os
import ensemble_model
from torchvision import transforms

def load_model_and_transform_from_checkpoint(checkpoint_path, tta):
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    args_ = checkpoint['args']
    if arch.startswith('Ensemble'):
        model = ensemble_model.__dict__[arch](multi_scale=args_.multi_scale if hasattr(args_, 'multi_scale') else False).cuda()
    else:    
        model = models.__dict__[arch](multi_scale=args_.multi_scale if hasattr(args_, 'multi_scale') else False).cuda()
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key[len('module.'):]] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    print(args_)
    if tta != 1:
        transform = build_transform(size=args_.img_size if hasattr(args_, 'img_size') else (496, 224), 
        rate_for_crop=args_.crop_rate, rand_filp=args_.flip, to='Tensor', norm=True, 
        updown_rand_crop=args_.updown_crop, translate=args_.translate, 
        canny=args_.canny if hasattr(args_, 'canny') else False, 
        hist=args_.hist, hist_level=args_.hist_level if hasattr(args_, 'hist_level') else 16, 
        colorjittor=args_.colorjittor if hasattr(args_, 'colorjittor') else [0,0,0,0])
    else:
        transform = build_transform(
            size=args_.img_size if hasattr(args_, 'img_size') else (496, 224), rate_for_crop=None, rand_filp=False, 
            to='Tensor', norm=True, 
            canny=args_.canny if hasattr(args_, 'canny') else False, 
            hist=args_.hist if hasattr(args_, 'hist') else False, 
            hist_level=args_.hist_level if hasattr(args_, 'hist_level') else 16)
    
    return model, checkpoint_path, transform, \
                args_.rzw if hasattr(args_, 'rzw') else [1,1], \
                args_.rzh if hasattr(args_, 'rzh') else [1,1], args_.rotate_prob




parser = argparse.ArgumentParser(description='debug_model')
parser.add_argument('--checkpoints', nargs='+', default=[''], type=str, metavar='PATH',
                    help='path of checkpoints')
parser.add_argument('--data_path', default='/data2/zrliu/dataset_cmp/dataset', type=str, help='data path')
parser.add_argument('--val', type=str, default='val', help='val dataset name.')
parser.add_argument('--split', type=str, default='37', help='split of dataset(37 or 46).')
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N',help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--softmean', action='store_true', help='whether to use softmax before mean ensemble.')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num_classes', type=int, default=8, help='number of classes.')
parser.add_argument('--mode', type=str, default='evaluate', help='mode.')
parser.add_argument('--tta', type=int, default=1, help='use same augmentation in training and val.')
parser.add_argument('--img_size', nargs=2, type=int, default=[496,224], help='image size')

def evaluate(args):
    model_list, name_list, transform_list, dataset_list, dataloader_list = [], [], [], [], []
    rzw, rzh, rotate_prob = [0,2], [0,2], 1
    for checkpoint in args.checkpoints:
        model, name, transform, rzw_, rzh_, rotate_prob_ = load_model_and_transform_from_checkpoint(checkpoint, args.tta)
        rzw[0] = max(rzw_[0], rzw[0])
        rzw[1] = min(rzw_[1], rzw[1])
        rzh[0] = max(rzh_[0], rzh[0])
        rzh[1] = min(rzh_[1], rzh[1])
        rotate_prob = min(rotate_prob_, rotate_prob)
        model_list.append(model)
        name_list.append(name)
        transform_list.append(transform)
        dataset = RoadAllData(args.data_path, transform=transform, mode=args.val, expand=False, split=args.split)
        dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)
    
    dataloader = dataloader_list[0]
    labels = []
    with torch.no_grad():
        scores_tta = []
        softmaxs_tta = []
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(len(dataloader), [batch_time, top1], prefix='Test: ')
        for t in range(args.tta):
            softmaxs_list, scores_list = [], []
            end = time.time()       
            for i, (images, target) in enumerate(dataloader):
                scores = []
                softmaxs = []
                for k, model in enumerate(model_list):
                    model.eval()
                    if args.tta != 1:
                        if rzw != [1,1] or rzh != [1,1]:
                            randw1 = int((np.random.rand() * (rzw[1] - rzw[0]) + rzw[0]) * args.img_size[1] + 0.5)
                            randh1 = int((np.random.rand() * (rzh[1] - rzh[0]) + rzh[0]) * args.img_size[0] + 0.5)
                            resize1 = transforms.Resize((randh1, randw1))
                            images = resize1(images)

                        if rotate_prob > np.random.rand():
                            images = images.permute(0, 1, 3, 2)
                    images = images.cuda()
                    target = target.cuda()
                    outs, _ = model(images)
                    scores.append(outs[0])
                    softmaxs.append(outs[0].softmax(dim=1))
                    if k == 0 and t == 0: labels.extend(target.detach().cpu().tolist())
                


                scores = sum(scores) / len(scores)
                softmaxs = sum(softmaxs) / len(softmaxs)
                scores_list.extend(scores.detach().cpu().tolist())
                softmaxs_list.extend(softmaxs.detach().cpu().tolist())

                acc1, = accuracy(outs[0], target[:, 0], topk=(1,))
                top1.update(acc1[0], images.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)   
            
            print("acc: ", top1.avg)
            scores_tta.append(torch.Tensor(scores_list))
            softmaxs_tta.append(torch.Tensor(softmaxs_list))
    
    labels = torch.Tensor(labels)
    if args.softmean:
        ps = sum(softmaxs_tta) / len(softmaxs_tta)
    else:
        ps = (sum(scores_tta) / len(scores_tta)).softmax(dim=1)

    preds = ps.argmax(dim=1).numpy()

    matrix = torch.zeros((2, args.num_classes))
    matrix[0, :] = labels[preds == 1, :args.num_classes].sum(axis=0)
    matrix[1, :] = labels[preds == 0, :args.num_classes].sum(axis=0)
    matrix /= matrix.sum(dim=0, keepdims=True)

    auc = roc_auc_score((1-labels[:, 0]).numpy().astype(int), ps[:, 0].numpy())

    acc = (torch.Tensor(preds) == labels[:, 0]).type(torch.float).mean() * 100
    print("acc :{:^2f}%".format(acc), "auc:{:^3f}".format(auc))
    print('\n\n')
    table = PrettyTable()
    table.title = 'result matrix'
    table.field_names = ['', '0', '1', '2', '3', '4', '5', '6', '7']
    table.add_row([0] + list(map(lambda x: round(x, 3), matrix.tolist()[0])))
    table.add_row([1] + list(map(lambda x: round(x, 3), matrix.tolist()[1])))
    print(table)

           

def submission(args):
    model_list, transform_list, dataset_list, dataloader_list = [], [], [], []
    rzw, rzh, rotate_prob = [0,2], [0,2], 1
    for checkpoint in args.checkpoints:
        model, _, transform, rzw_, rzh_, rotate_prob_ = load_model_and_transform_from_checkpoint(checkpoint, args.tta)
        rzw[0] = max(rzw_[0], rzw[0])
        rzw[1] = min(rzw_[1], rzw[1])
        rzh[0] = max(rzh_[0], rzh[0])
        rzh[1] = min(rzh_[1], rzh[1])
        rotate_prob = min(rotate_prob_, rotate_prob)
        model_list.append(model)
        transform_list.append(transform)
        dataset = RoadAllDataTest(data_path=os.path.join(args.data_path, 'test_images'), transform=transform)
        dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)
    
    dataloader = dataloader_list[0]
    print("rzw: \t\t", rzw)
    print("rzh: \t\t", rzh)
    print("rotate: \t", rotate_prob)


    
    with torch.no_grad():
        
        if args.softmean:   softmaxs_tta = []
        else:   scores_tta = []
        for t in range(args.tta):
            name_list = []
            if args.softmean:   softmaxs_list = []
            else:   scores_list = []        
            for images, names in tqdm(dataloader):
                
                if args.softmean:   softmaxs = []
                else:   scores = []
                for k, model in enumerate(model_list):
                    model.eval()
                    if args.tta != 1:
                        if rzw != [1,1] or rzh != [1,1]:
                            randw1 = int((np.random.rand() * (rzw[1] - rzw[0]) + rzw[0]) * args.img_size[1] + 0.5)
                            randh1 = int((np.random.rand() * (rzh[1] - rzh[0]) + rzh[0]) * args.img_size[0] + 0.5)
                            resize1 = transforms.Resize((randh1, randw1))
                            images = resize1(images)

                        if rotate_prob > np.random.rand():
                            images = images.permute(0, 1, 3, 2)
                    images = images.cuda()
                    outs, _ = model(images)
                    
                    if args.softmean:   softmaxs.append(outs[0].softmax(dim=1))
                    else:   scores.append(outs[0])
                    if k == 0 and t == 0: name_list.extend(names)
                
                
                if args.softmean:   softmaxs = sum(softmaxs) / len(softmaxs)
                else:   scores = sum(scores) / len(scores)
                
                if args.softmean:   softmaxs_list.extend(softmaxs.detach().cpu().tolist())
                else:   scores_list.extend(scores.detach().cpu().tolist())
            
            if args.softmean:   softmaxs_tta.append(torch.Tensor(softmaxs_list))
            else:   scores_tta.append(torch.Tensor(scores_list))

    if args.softmean:
        ps = sum(softmaxs_tta) / len(softmaxs_tta)
    else:
        ps = (sum(scores_tta) / len(scores_tta)).softmax(dim=1)

    preds = ps[:, 0].tolist()
    submission = pd.DataFrame({'imagename':name_list, 'defect_prob':preds})
    submission.to_csv('./submission.csv', index=False, encoding='utf-8')

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


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'submission':
        submission(args)

