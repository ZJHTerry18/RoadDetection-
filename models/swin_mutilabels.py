from re import L
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.swin_transformer import Swin_B_Weights, Swin_T_Weights, Swin_S_Weights, Swin_V2_B_Weights
from .swin_tranformer_v2 import SwinTransformerV2, swin_v2_b_backbone

__all__ = ['SwinModel', 'SwinBModel', 'SwinTModel', 'SwinV2BModel', 'SwinV2B_ori']

class SwinSModel(nn.Module):
    def __init__(self, pretrained=False, auxiliary=False, num_classes=8, multi_cls=False, multi_scale=False):
        super(SwinSModel, self).__init__()
        assert not auxiliary, 'auxiliary is not supported in Swin-S model.'
        self.num_classes = num_classes
        self.backbone = models.swin_s(weights=(Swin_S_Weights.IMAGENET1K_V1 if pretrained else None))
        self.backbone.head = torch.nn.Sequential()
        self.fc = nn.ModuleList([nn.Linear(768, 2) for _ in range(self.num_classes)])

        
    
    def forward(self, x):
        feature = self.backbone(x)
        
        outs = []
        for i in range(self.num_classes):
           outs.append(self.fc[i](feature))

        return outs, feature


class SwinBModel(nn.Module):
    def __init__(self, pretrained=False, auxiliary = False, num_classes=8, multi_cls=False, multi_scale=False):
        super(SwinBModel, self).__init__()
        assert not auxiliary, 'auxiliary is not supported in Swin-B model.'
        self.num_classes = num_classes
        self.backbone = models.swin_b(weights=(Swin_B_Weights.IMAGENET1K_V1 if pretrained else None))
        self.backbone.head = torch.nn.Sequential()
        self.fc = nn.ModuleList([nn.Linear(1024, 2) for _ in range(self.num_classes)])
        
    
    def forward(self, x):
        feature = self.backbone(x)
        
        outs = []
        for i in range(self.num_classes):
           outs.append(self.fc[i](feature))

        return outs, feature


class SwinTModel(nn.Module):
    def __init__(self, pretrained=False, auxiliary = False, num_classes=8, multi_cls=False, multi_scale=False):
        super(SwinTModel, self).__init__()
        assert not auxiliary, 'auxiliary is not supported in Swin-T model.'
        self.num_classes = num_classes
        self.backbone = models.swin_t(weights=(Swin_T_Weights.IMAGENET1K_V1 if pretrained else None))
        self.backbone.head = torch.nn.Sequential()
        self.fc = nn.ModuleList([nn.Linear(768, 2) for _ in range(self.num_classes)])


    
    def forward(self, x):
        feature = self.backbone(x)
        
        outs = []
        for i in range(self.num_classes):
           outs.append(self.fc[i](feature))

        return outs, feature


class SwinV2BModel(nn.Module):
    def __init__(self, pretrained=False, auxiliary = False, num_classes=8, multi_cls=False, multi_scale=False):
        super(SwinV2BModel, self).__init__()
        assert not auxiliary, 'auxiliary is not supported in Swin-T model.'
        self.num_classes = num_classes
        self.backbone = models.swin_v2_b(weights=(Swin_V2_B_Weights.IMAGENET1K_V1 if not pretrained else None))
        self.backbone.head = torch.nn.Sequential()
        self.fc = nn.ModuleList([nn.Linear(1024, 2) for _ in range(self.num_classes)])


    
    def forward(self, x):
        feature = self.backbone(x)
        
        outs = []
        for i in range(self.num_classes):
           outs.append(self.fc[i](feature))

        return outs, feature


class SwinV2B_ori(nn.Module):
    def __init__(self, pretrained=False, auxiliary = False, num_classes=8, multi_cls=False, multi_scale=False):
        super(SwinV2B_ori, self).__init__()
        assert not auxiliary, 'auxiliary is not supported in Swin-T model.'
        self.num_classes = num_classes
        self.backbone = swin_v2_b_backbone(pretrained=pretrained)
        self.backbone.head = nn.Identity()
        self.fc = nn.ModuleList([nn.Linear(1024, 2) for _ in range(self.num_classes)])


    
    def forward(self, x):
        feature = self.backbone(x)
        
        outs = []
        for i in range(self.num_classes):
           outs.append(self.fc[i](feature))

        return outs, feature


def my_swin_s(**kwargs):
    return SwinSModel(**kwargs)

def my_swin_b(**kwargs):
    return SwinBModel(**kwargs)

def my_swin_t(**kwargs):
    return SwinTModel(**kwargs)

def my_swin_v2_b(**kwargs):
    return SwinV2BModel(**kwargs)

def my_swin_v2_b_ori(**kwargs):
    return SwinV2B_ori(**kwargs)