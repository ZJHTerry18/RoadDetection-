import torch.utils.data as Data
import numpy as np
import torch
import PIL
import os
import pandas as pd
import glob

class RoadAllData(Data.Dataset):
    def __init__(self, dataset_path, transform=None, sepcial_transform=None, sepcial_prob=0.3, two_transforms=False, load_labeled=True, load_unlabeled=False, mode='train', expand=True, scale_length=1, num_classes=8, split='37'):
        super(RoadAllData, self).__init__()
        self.load_unlabeled = load_unlabeled
        self.scale_length = scale_length
        self.two_transforms = two_transforms
        self.transform = transform
        self.sepcial_transform = sepcial_transform
        self.sepcial_prob = sepcial_prob
        self.sepcial_transform_categories = ['roadsh', 'roadbr', 'white', 'yellong', 'stop', 'blue']
        self.num_classes = num_classes
        self.mode = mode
        self.img_list = []


        # 来锻炼一下大家的逻辑能力，这堆代码我就不加注释了 (﹁ ﹁) ~→
        assert not (two_transforms and (mode == 'val')) or (two_transforms and (mode == 'val1')) or (two_transforms and (mode == 'uniform_val'))
        # assert (scale_length == 1) or (mode == 'train')
        assert (mode == 'train') or (mode == 'val' and not expand) or (mode == 'val1' and not expand) or (mode == 'uniform_val' and not expand)
        assert split in ['37', '46']
        assert split == '37' or not expand
        assert split == '37' or ((split == '46') and (mode in ['train', 'val']))

        csv_path = os.path.join(dataset_path, 'train_label', ('hard_' if split == '46' else '') + mode +('_expand' if expand else '' )+'.csv')
        label_df = pd.read_csv(csv_path, encoding='utf-8', sep='\t', header=None).values.tolist()
        
        label_df = [l[0].split(',') for l in label_df]
        if load_labeled:
            self.img_list = [l[0] for l in label_df]
            if mode == 'uniform_val':
                self.img_list = list(map(lambda x: os.path.join(dataset_path, 'test_images', x), self.img_list))
            self.img_list = list(map(lambda x: os.path.join(dataset_path, 'train_image/labeled_data', x), self.img_list))
        
            ## labeled
            labels = torch.zeros((len(self.img_list), 9)).long()
            for idx in range(len(self.img_list)):
                labeldata = label_df[idx][1:]
                assert len(labeldata) > 0, '有无标签的数据！'
                for i in labeldata:
                    if i != '':
                        i = int(i)
                        labels[idx][i] = 1
        ## unlabeled
        if load_unlabeled:
            unlabel_path = os.path.join(dataset_path, 'train_image/unlabeled_data')
            unlabeled_names = os.listdir(unlabel_path)
            unlabeled_names = list(map(lambda x: os.path.join(unlabel_path, x), unlabeled_names))
            self.img_list.extend(unlabeled_names)
            unlabeled_labels = torch.zeros((len(unlabeled_names), 9)).long()
            unlabeled_labels[:, 8] = torch.arange(len(unlabeled_names)) + 8
            
            self.label_list = torch.concat([labels, unlabeled_labels], dim=0) if load_labeled else unlabeled_labels
        else:
            self.label_list = labels
        
        

    def __getitem__(self, idx):
        imagename = self.img_list[idx]
        image = PIL.Image.open(imagename).convert('RGB')

        label = self.label_list[idx].clone()

         
        prob_num, prob = np.random.randint(1, self.num_classes-1), np.random.rand()
        if self.two_transforms:
            image, label = self.mytransform(image, label, prob_num=prob_num, prob=prob)
            image1 = self.transform(image)
            image2 = self.transform(image)      
            return (image1, image2), label
        else:
            image, label = self.mytransform(image, label, prob_num=prob_num, prob=prob)
            return image, label
            
    def mytransform(self, image, label, prob_num, prob):
        if self.sepcial_transform != None:
            if (label[0] != 0 or label[self.num_classes] != 0) and (prob < self.sepcial_prob):
                image = self.sepcial_transform[self.sepcial_transform_categories[prob_num-1]](image)
                label[prob_num] = 1
                label[0] = 0
                label[self.num_classes] = 0
        if self.transform and (not self.two_transforms):
            image = self.transform(image)
        return image, label

    def __len__(self):
        return int(len(self.img_list) * self.scale_length)

class RoadAllDataTest(Data.Dataset):
    def __init__(self, data_path, transform, scale_length=1):
        super(RoadAllDataTest, self).__init__()
        self.img_list = glob.glob(os.path.join(data_path, '*.png'))
        self.transform = transform
        self.scale_length = scale_length
    
    def __getitem__(self, idx):
        img = PIL.Image.open(self.img_list[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.img_list[idx])

    def __len__(self):
        return int(len(self.img_list) * self.scale_length)
    
    def sort(self):
        self.img_list.sort(key=lambda x: int(os.path.basename(x)[:-4].split('_')[1]))