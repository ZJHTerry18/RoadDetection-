from ctypes import sizeof
import torch
import numpy as np
import models
from data import build_transform, RoadAllData, RoadAllDataTest, Canny
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import argparse
from torchvision import transforms
import cv2


def load_model_and_transform_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    args_ = checkpoint['args']
    model = models.__dict__[arch](multi_cls=args_.multi_cls if hasattr(args_, 'multi_cls') else False,
                        multi_scale=args_.multi_scale if hasattr(args_, 'multi_scale') else False).cuda()
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key[len('module.'):]] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    print(args_)
    transform = build_transform(
        size=args_.img_size if hasattr(args_, 'img_size') else (496, 224), rate_for_crop=None, rand_filp=False, 
        to='Tensor', norm=True, 
        canny=args_.canny if hasattr(args_, 'canny') else False, 
        hist=args_.hist if hasattr(args_, 'hist') else False, 
        hist_level=args_.hist_level if hasattr(args_, 'hist_level') else 16)

    return model, arch, transform

canny = Canny()

class DebugModel(torch.nn.Module):
    def __init__(self, checkpoints_path_list, model_name_list, data_path, split='37', mode='val', num_classes=8, workers=12, batch_size=2, scale_length=1, scale=0.25):
        self.models, self.archs, self.transforms, self.dataloaders = [], [], [], []
        for checkpoint_path in checkpoints_path_list:
            model, arch, transform = load_model_and_transform_from_checkpoint(checkpoint_path)
            self.models.append(model)
            self.archs.append(arch)
            self.transforms.append(transform)
            
        self.mode = mode
        self.num_classes = num_classes
        self.model_name_list = model_name_list
        self.archs = self.model_name_list # hack
        self.data_path = data_path
        self.split = split
        self.ToPILImage = transforms.ToPILImage()
        self.scale = scale
        
        for transform in self.transforms:
            dataset = self.build_dataset(transform=transform, scale_length=scale_length)
            dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=1, shuffle=False,
                            num_workers=workers, pin_memory=True)
            self.dataloaders.append(dataloader)
        
        self.dataset_naive = self.build_dataset(transform=None, scale_length=scale_length)

    def build_dataset(self, transform, scale_length):
        if self.mode == 'test':
            dataset = RoadAllDataTest(data_path=os.path.join(self.data_path, 'test_images'), 
                                            transform=transform, scale_length=scale_length)
            dataset.sort()
        else:
            dataset = RoadAllData(dataset_path=self.data_path, 
                                        transform=transform, 
                                        mode=self.mode, 
                                        expand=False, 
                                        num_classes=self.num_classes, split=self.split, scale_length=scale_length)
        
        return dataset

    def inference_val(self):
        self.scores, self.preds, self.labels = {}, {}, []
        for key in self.archs:
            self.scores[key] = []
            self.preds[key] = []
        for model in self.models:
            model.eval()
           
        with torch.no_grad():
            for k, (model, arch, dataloader) in enumerate(zip(self.models, self.archs, self.dataloaders)):
                print('=> inference using model {}'.format(arch))
                for imgs, labels in tqdm(dataloader):
                    imgs = imgs.cuda()
                    outs, feat = model(imgs)
                    out = outs[0]
                    out0 = out[:, 1]
                    out1 = out[:, 0]
                    out = torch.flip(out, dims=[1]).detach().cpu().numpy().tolist()
                    self.scores[arch].extend(out)
                    self.preds[arch].extend((out0 < out1).detach().cpu().numpy().astype(int).tolist())

                    if k == 0:
                        self.labels.extend(labels.numpy().tolist())

    def inference_test(self):
        self.scores, self.preds = {}, {}
        for key in self.archs:
            self.scores[key] = []
            self.preds[key] = []
        for model in self.models:
            model.eval()
            
        for k, (model, arch, dataloader) in enumerate(zip(self.models, self.archs, self.dataloaders)):
                print('=> inference using model {}'.format(arch))
                for imgs, names in tqdm(dataloader):
                    imgs = imgs.cuda()
                    outs, feat = model(imgs)
                    out = outs[0]
                    out0 = out[:, 1]
                    out1 = out[:, 0]
                    out = torch.flip(out, dims=[1]).detach().cpu().numpy().tolist()
                    self.scores[arch].extend(out)
                    self.preds[arch].extend((out0 < out1).detach().cpu().numpy().astype(int).tolist())

                
    def display_val(self, display_mode):
        if display_mode.startswith('pred'):
            target = int(display_mode[len('pred'):])
        elif display_mode.startswith('fault'):
            if display_mode != 'fault':
                target = int(display_mode[len('fault'):])
        elif display_mode.startswith('all'):
            if display_mode != 'all':
                target = int(display_mode[len('all'):])
        elif display_mode.startswith('diff'):
            if display_mode != 'diff':
                target = int(display_mode[len('diff'):])

        for i in range(len(self.dataset_naive)):
            if display_mode.startswith('pred'):
                flag = False
                for arch in self.archs:
                    if self.preds[arch][i] == target:
                        flag = True
                        break
                if not flag:    continue
            elif display_mode == 'fault':
                flag = False
                for arch in self.archs:
                    if self.preds[arch][i] == self.labels[i][0]:
                        flag = True
                if not flag:    continue
            elif display_mode.startswith('fault'):
                flag = False
                for arch in self.archs:
                    if self.preds[arch][i] == self.labels[i][0] and self.labels[i][target]:
                        flag = True
                if not flag:    continue
            elif display_mode == 'diff':
                if self.preds[self.archs[0]][i] == self.preds[self.archs[1]][i]: continue 
            elif display_mode.startswith('diff'):
                if self.preds[self.archs[0]][i] == self.preds[self.archs[1]][i] or not self.labels[i][target]: continue
            elif display_mode.startswith('all') and display_mode != 'all':
                if not self.labels[i][target]: continue

            img, _ = self.dataset_naive[i]
            print('\n\n\n\n\n')
            table = PrettyTable()
            label = ''
            for j, l in enumerate(self.labels[i]):
                if l:   label += ' {}'.format(j)
            table.title = os.path.basename(self.dataset_naive.img_list[i]) + label 
            table.field_names = ['model', 'score', 'p', 'pred']
            for model_name, arch in zip(self.model_name_list, self.archs):
                table.add_row([model_name, 
                    self.format(self.scores[arch][i]), 
                    self.format(self.scores[arch][i], softmax=True), 
                    self.preds[arch][i]])
            print(table)
            img = np.array(img).astype('uint8')
            img = self.showImage(img)
            img = cv2.resize(img, (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            while True:
                cv2.namedWindow(table.title)
                cv2.moveWindow(table.title, 0, 0)
                cv2.imshow(table.title, img)
                cv2.resizeWindow(table.title, img.shape[1], img.shape[0])
                key = cv2.waitKey()
                cv2.destroyWindow(table.title)
                if key == ord('q'):
                    return
                elif key == ord('r'):
                    continue
                else:
                    break


    def display_test(self, display_mode):
        if display_mode.startswith('pred'):
            target = int(display_mode[len('pred'):])
        for i in range(len(self.dataset_naive)):
            
            if display_mode.startswith('pred'):
                flag = False
                for arch in self.archs:
                    if self.preds[arch][i] == target:
                        flag = True
                        break
                if not flag:    continue
            elif display_mode == 'diff':
                if self.preds[self.archs[0]][i] == self.preds[self.archs[1]][i]: continue 


            img, name = self.dataset_naive[i]
            print('\n\n\n\n\n')
            table = PrettyTable()
            table.title = name
            table.field_names = ['model', 'score', 'p', 'pred']
            for model_name, arch in zip(self.model_name_list, self.archs):
                table.add_row([model_name, 
                    self.format(self.scores[arch][i]), 
                    self.format(self.scores[arch][i], softmax=True), 
                    self.preds[arch][i]])
            print(table)
            img = np.array(img).astype('uint8')
            img = self.showImage(img)
            img = cv2.resize(img, (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            while True:
                cv2.namedWindow(table.title)
                cv2.moveWindow(table.title, 0, 0)
                cv2.imshow(table.title, img)
                cv2.resizeWindow(table.title, img.shape[1], img.shape[0])
                key = cv2.waitKey()
                cv2.destroyWindow(table.title)
                if key == ord('q'):
                    return
                elif key == ord('r'):
                    continue
                else:
                    break


    def format(self, x, softmax=False):
        if softmax: x = torch.Tensor(x).softmax(dim=0).tolist()
        return "{:^6f}  {:^6f}".format(x[0], x[1])

    def showImage(self, img):
        cannyimg = canny(img)
        white = np.full((img.shape[0], 200, 3), 255).astype('uint8')
        return np.concatenate([img, white, cannyimg], axis=1)



parser = argparse.ArgumentParser(description='debug_model')
parser.add_argument('--checkpoints', nargs='+', default=[''], type=str, metavar='PATH',
                    help='path of checkpoints')
parser.add_argument('--names', nargs='+', default=[''], type=str, help='checkpoint names.')
parser.add_argument('--data_path', default='/data2/zrliu/dataset_cmp/dataset', type=str, help='data path')
parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N',help='mini-batch size (default: 2)')
parser.add_argument('--split', type=str, default='37', help='split of dataset(37 or 46).')
parser.add_argument('--mode', type=str, default='test', help='val or test.')
parser.add_argument('--scale_length', type=float, default=1, help="scale torch dataset's length.")
parser.add_argument('--display_mode', type=str, default='all', help='display mode.')
parser.add_argument('--scale', type=float, default=0.25, help='image rescale factor.')
parser.add_argument('--hist', action='store_true', help='whether to use hist mode.')
args = parser.parse_args()

if __name__ == '__main__':
    debugmodel = DebugModel(args.checkpoints, args.names, args.data_path, args.split, args.mode, scale_length=args.scale_length, scale=args.scale)
    if args.mode == 'test':
        debugmodel.inference_test()
        while True:
            args.display_mode = input('please enter display mode: ')
            if args.display_mode == 'exit': break
            if args.display_mode not in ['all', 'pred1', 'pred0', 'diff']: 
                print('unrecognition mode.')                
                continue
            if args.display_mode.startswith('diff') and len(args.names) != 2:   
                print('mode diff need have and only have two models.')
                continue   
            debugmodel.display_test(args.display_mode)
            
    else:
        debugmodel.inference_val()
        while True:
            args.display_mode = input('please enter display mode: ')
            if args.display_mode == 'exit': break
            if args.display_mode not in ['all', 'all0', 'all1', 'all2', 'all3', 
                                    'all4', 'all5', 'all6', 'all7', 
                                'fault', 'fault1', 'fault2', 'fault3', 
                                'fault4', 'fault5', 'fault6', 'fault7', 'pred0', 'pred1', 
                                'diff', 'diff0', 'diff1', 'diff2', 'diff3', 'diff4', 'diff5', 'diff6', 'diff7']: 
                print('unrecognition mode.')                
                continue
            if args.display_mode.startswith('diff') and len(args.names) != 2:   
                print('mode diff need have and only have two models.')
                continue    
            debugmodel.display_val(args.display_mode)






'''
DebugModel:
1. 测试集显示图像，输出 预测类别 | 得分 | 标签；
2. 验证集显示图像，输出[预测错误] 预测类别 | 得分 | 标签；
3. 多个模型，进行输出：
    两个模型预测不同的集合
    预测错误的集合
'''



'''
得分大于 某个值 的正样本比例
'''