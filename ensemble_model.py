from turtle import forward
import models
import torch
import torch.nn.functional as F

__all__ = ['EnsembleDeepFuseModel', 'EnsembleModel']


class EnsembleDeepFuseModel(torch.nn.Module):
    def __init__(self, archs, pretrained, auxiliary, num_classes, multi_cls, multi_scale):
        super(EnsembleDeepFuseModel, self).__init__()
        self.num_classes = num_classes
        self.auxiliary = auxiliary

        self.classifiers = []
        for arch in archs:
            self.classifiers.append(models.__dict__[arch](pretrained=pretrained, auxiliary=False, num_classes=num_classes, multi_cls=multi_cls, multi_scale=multi_scale)) 
        
        self.classifiers = torch.nn.ModuleList(self.classifiers)
        total_dim = 0
        for classifier in self.classifiers:
            total_dim += classifier.fc[0].weight.shape[1]
        self.fuse_classifier = torch.nn.ModuleList([torch.nn.Linear(total_dim, 2) for _ in range(self.num_classes)])

    def forward(self, x):
        outs, feats = [[] for i in range(self.num_classes)], []
        for i in range(len(self.classifiers)):
            outs_, feats_ = self.classifiers[i](x)
            for j in range(self.num_classes):
                outs[j].append(outs_[j])
            feats.append(feats_)
        
        cat_feats = torch.cat(feats, dim=1)
        for i in range(self.num_classes):
            outs[i].append(self.fuse_classifier[i](cat_feats))
            outs[i] = sum(outs[i]) / len(outs[i])

        if self.training and self.auxiliary:
            return outs, [F.normalize(cat_feats, dim=1)]
        return outs, cat_feats

    def load_a_state_dict(self, state_dict, idx, strict=False):
        msg = self.classifiers[idx].load_state_dict(state_dict, strict=strict)
        return msg

class EnsembleModel(torch.nn.Module):
    def __init__(self, archs, pretrained, auxiliary, num_classes, multi_cls, multi_scale):
        super(EnsembleModel, self).__init__()
        self.num_classes = num_classes
        self.auxiliary = auxiliary

        self.classifiers = []
        for arch in archs:
            self.classifiers.append(models.__dict__[arch](pretrained=pretrained, auxiliary=False, num_classes=num_classes, multi_cls=multi_cls, multi_scale=multi_scale)) 
        
        self.classifiers = torch.nn.ModuleList(self.classifiers)
        
    def forward(self, x):
        outs, feats = [[] for i in range(self.num_classes)], []
        for i in range(len(self.classifiers)):
            outs_, feats_ = self.classifiers[i](x)
            for j in range(self.num_classes):
                outs[j].append(outs_[j])
            feats.append(feats_)
        
        cat_feats = torch.cat(feats, dim=1)
        for i in range(self.num_classes):
            outs[i] = sum(outs[i]) / len(outs[i])

        if self.training and self.auxiliary:
            return outs, [F.normalize(cat_feats, dim=1)]
        return outs, cat_feats

    def load_a_state_dict(self, state_dict, idx, strict=False):
        msg = self.classifiers[idx].load_state_dict(state_dict, strict=strict)
        return msg