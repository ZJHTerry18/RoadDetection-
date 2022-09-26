# coding=utf-8
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import models
import argparse
from data import RoadAllData, build_transform, build_special_transform
from torchvision import transforms

parser = argparse.ArgumentParser(description='submission')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='checkpoint')
parser.add_argument('--data_path', default='/data2/zrliu/dataset_cmp/dataset', type=str, help='data path')
parser.add_argument('--layer_name', default='layer3', help='layer name')
parser.add_argument('--dataset', default='val', type=str, help='train dataset or test dataset.')
parser.add_argument('--load_unlabeled', action='store_true', help='load unlabeled dataset.')
parser.add_argument('--special_transform', action='store_true', help='whether to use special transform.')
parser.add_argument('--special_prob', type=float, default=0.3, help='probability of performing a special transformation on an image.')
parser.add_argument('--updown_crop', type=float, nargs='+', default=[0, 0, 0], help='Probability of top, undertop, bottom crop')
parser.add_argument('--old_version', action='store_true', help='to support old checkpoint.')

class GradCAM:
    def __init__(self, model: nn.Module, size=(392, 184), num_cls=1000, mean=None, std=None, layer_name='layer3') -> None:
        self.model = model
        self.model.eval()
        layer_dict = {'layer1' : self.model.layer1, 
                        'layer2' : self.model.layer2, 
                        'layer3' : self.model.layer3, 
                        'layer4' : self.model.layer4
                        }
        #model.后面，register前面是要反馈的那层网络的名字
        layer_dict[layer_name].register_forward_hook(self.__forward_hook)
        if int((torch.__version__).split('+')[0].split('.')[1]) == 7:
            layer_dict[layer_name].register_backward_hook(self.__backward_hook)
        else:
            layer_dict[layer_name].register_full_backward_hook(self.__backward_hook)
        self.size = size
        self.origin_size = size
        self.num_cls = num_cls

        self.grads = []
        self.fmaps = []

    def forward(self, img_arr: np.ndarray, label_str, label=None, show=True, write=False):
        img_input1 = self.__img_preprocess(img_arr.copy())
        img_input = torch.tensor(img_input1)
        img_input = Variable(img_input.cuda())
        # forward
        output = self.model(img_input)
        output = output[0]
        idx = np.argmax(output.cpu().data.numpy())#最大值对应的索引



        # backward
        self.model.zero_grad()
        loss = self.__compute_loss(output, label)

        loss.backward()
        #print("self.grads:", self.grads)
        # generate CAM
        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads_val)

        # show
        cam_show = cv2.resize(cam, self.origin_size)
        img_show = img_arr.astype(np.float32) / 255
        self.__show_cam_on_image(img_show, cam_show, label_str, if_show=show, if_write=write)

        self.fmaps.clear()
        self.grads.clear()

    def __img_transform(self, img_arr: np.ndarray, transform: torchvision.transforms) -> torch.Tensor:
        img = img_arr.copy()  # [H, W, C]
        img = Image.fromarray(np.uint8(img))
        img = transform(img).unsqueeze(0)  # [N,C,H,W]
        return img

    #图片处理
    def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
        #self.origin_size = (img_in.shape[1], img_in.shape[0])  # [H, W, C]
        img = img_in.copy()
        #img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        transform = transforms.Compose([
            transforms.ToTensor(),

        ])
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())



    def __forward_hook(self, module, input, output):
        self.fmaps.append(output)


    #算损失函数用来反馈
    def __compute_loss(self, logit, index=None):
        if not index:
            index = np.argmax(logit.cpu().data.numpy())
        else:
            index = np.array(index)

        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = 0.001*torch.zeros(1, self.num_cls).scatter_(1, index, 1).cuda()
        one_hot.requires_grad = True
        loss = torch.sum(one_hot * logit)
        return loss

    #计算cam图
    def __compute_cam(self, feature_map, grads):
        """
        feature_map: np.array [C, H, W]
        grads: np.array, [C, H, W]
        return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        alpha = np.mean(grads, axis=(1, 2))  # GAP
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]  # linear combination

        cam = np.maximum(cam, 0)  # relu
        cam = cv2.resize(cam, self.size)
        cam = (cam - np.min(cam)) / np.max(cam)
        return cam

    #保存cam图片
    def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, label: str, if_show=True, if_write=False):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)*0.5 
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        if if_write:
            cv2.imwrite("camcam.jpg", cam)
        if if_show:
            # 要显示RGB的图片，如果是BGR的 热力图是反过来的
            cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
            cv2.imshow(label, cam)
            
            key = cv2.waitKey()
            cv2.destroyAllWindows()
            if key == ord('q'):
                exit()
            



def main():
    args = parser.parse_args()
    transform = build_transform(norm=False, to='ndarray', updown_rand_crop=args.updown_crop)
    
    if args.special_transform:
        special_transform = build_special_transform()
    else:
        special_transform = None
    
    dataset = RoadAllData(args.data_path, transform=transform, 
                            sepcial_transform=special_transform,
                            sepcial_prob=args.special_prob,
                            two_transforms=False, 
                            load_unlabeled=args.load_unlabeled, 
                            mode=args.dataset, 
                            expand=False)
    
    ### load model
    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        arch = checkpoint['arch']
    else:
        state_dict = checkpoint
        arch = 'resnet18'
    # delete renamed or unused k    
    for k in list(state_dict.keys()):
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]

    if args.old_version:    
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]
    model = models.__dict__[arch]().cuda()
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    for i in range(len(dataset)):
        img, label = dataset[i]
        labels = []
        for j in range(8):
            if label[j]:
                labels.append(str(j))
        if len(labels) == 0:
            label_str = 'unlabeled'
        else:
            label_str = ','.join(labels)
        #model就是我们的网络名字，'base_resnet'不用管，在后面init里面改反馈的层比较方便
        #(2340, 1080)是图片的resize大小，其余俩是用于归一化的（咱们网络不需要这个）
        grad_cam = GradCAM(model, size=(img.shape[1], img.shape[0]), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], num_cls=2, layer_name=args.layer_name)
        
        grad_cam.forward(img, label_str, 0, show=True, write=False)

if __name__ == '__main__':
    main()