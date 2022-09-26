from .resnet_mutilabels import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from .swin_mutilabels import my_swin_b, my_swin_s, my_swin_t, my_swin_v2_b, my_swin_v2_b_ori
from .efficientnet_multilabels import eff_b7, eff_v2_s, eff_v2_m, eff_v2_l
from .hist_mlp import hist_mlp_16, hist_mlp_64, hist_mlp
from torch import nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'my_swin_s', 'my_swin_t', 'my_swin_b', 'my_swin_v2_b', 'my_swin_v2_b_ori',
           'eff_b7', 'eff_v2_s', 'eff_v2_m', 'eff_v2_l', 'hist_mlp_16', 'hist_mlp_64', 'hist_mlp']




