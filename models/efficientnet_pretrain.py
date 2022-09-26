from re import L
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B7_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
import torch.nn.functional as F

__all__ = ['EffB7Model', 'EffV2SModel', 'EffV2MModel', 'EffV2LModel']

class EffB7Model(nn.Module):
    def __init__(self, pretrained=False, auxiliary=False, num_classes=8, multi_cls=False, multi_scale=False):
        super(EffB7Model, self).__init__()
        # assert not auxiliary, 'auxiliary is not supported in efficientnet model.'
        self.num_classes = num_classes
        self.auxiliary = auxiliary
        self.backbone = models.efficientnet_b7(weights=(EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None))
        self.backbone.classifier = torch.nn.Sequential()
        self.fc = nn.Linear(2560, 2)

        
    
    def forward(self, x):
        feature = self.backbone(x)
        
        return self.fc(feature)


class EffV2SModel(nn.Module):
    def __init__(self, pretrained=False, auxiliary=False, num_classes=8, multi_cls=False, multi_scale=False):
        super(EffV2SModel, self).__init__()
        # assert not auxiliary, 'auxiliary is not supported in efficientnet model.'
        self.auxiliary = auxiliary
        self.num_classes = num_classes
        self.backbone = models.efficientnet_v2_s(weights=(EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None))
        self.backbone.classifier = torch.nn.Sequential()
        self.fc = nn.Linear(1280, 2)

        
    
    def forward(self, x):
        feature = self.backbone(x)
        
        return self.fc(feature)


class EffV2MModel(nn.Module):
    def __init__(self, pretrained=False, auxiliary=False, num_classes=8, multi_cls=False, multi_scale=False):
        super(EffV2MModel, self).__init__()
        # assert not auxiliary, 'auxiliary is not supported in efficientnet model.'
        self.num_classes = num_classes
        self.backbone = models.efficientnet_v2_m(weights=(EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None))
        self.backbone.classifier = torch.nn.Sequential()
        self.fc = nn.Linear(1280, 2)

        
    
    def forward(self, x):
        feature = self.backbone(x)
        
        return self.fc(feature)


class EffV2LModel(nn.Module):
    def __init__(self, pretrained=False, auxiliary=False, num_classes=8, multi_cls=False, multi_scale=False):
        super(EffV2LModel, self).__init__()
        # assert not auxiliary, 'auxiliary is not supported in efficientnet model.'
        self.num_classes = num_classes
        self.backbone = models.efficientnet_v2_l(weights=(EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None))
        self.backbone.classifier = torch.nn.Sequential()
        self.fc = nn.Linear(1280, 2)

        
    
    def forward(self, x):
        feature = self.backbone(x)
        
        return self.fc(feature)

def eff_b7(**kwargs):
    return EffB7Model(**kwargs)

def eff_v2_s(**kwargs):
    return EffV2SModel(**kwargs)

def eff_v2_m(**kwargs):
    return EffV2MModel(**kwargs)

def eff_v2_l(**kwargs):
    return EffV2LModel(**kwargs)
