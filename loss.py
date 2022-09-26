from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()

class MultiBCEloss(nn.Module):
    def __init__(self,num_classes=8):
        super(MultiBCEloss, self).__init__()
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss()
    
    def forward(self, inputs, targets):
        preds = self.sigmoid(inputs)
        lossvalue = self.bceloss(preds, targets)

        return lossvalue


class maskedMultiCEloss(nn.Module):
    def __init__(self,num_classes=8, weight=False):
        super(maskedMultiCEloss, self).__init__()
        self.nums = [25767, 231, 778, 243, 366, 324, 1936, 329]
        self.sum_nums = sum(self.nums)
        self.num_classes = num_classes
        self.criterion = []
        for i in range(num_classes):
            if weight:
                print("=> weight enabled.")
                self.criterion.append(nn.CrossEntropyLoss(weight=torch.Tensor([self.nums[i] / self.sum_nums, 1 - self.nums[i] / self.sum_nums])))
            else:
                self.criterion.append(nn.CrossEntropyLoss())
        self.criterion = nn.ModuleList(self.criterion)
    def forward(self, preds, labels, mask):
        loss = 0.0
        for i in range(self.num_classes):
            preds_cls = preds[i][mask, :]
            labels_cls = labels[mask, i]
            lossvalue = self.criterion[i](preds_cls, labels_cls)
            loss += lossvalue / self.num_classes
    
        return loss


class multiCEloss(nn.Module):
    def __init__(self, args, num_classes=8, labelsmooth=0.0, alpha=[0.5, 0.5], gamma=0.0):
        super(multiCEloss, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.multi_cls = args.multi_cls
        self.multi_pos = args.multi_pos
        self.bce = args.bce
        self.alpha = [args.alpha[0]] + [args.alpha[1]] * (args.num_classes - 1)
        self.gamma = args.gamma
        self.criterion = CrossEntropyLabelSmooth(epsilon=args.labelsmooth)
        self.multiad_criterion = MultiPositiveLoss(num_classes=args.num_classes)
        self.bce_criterion = MultiBCEloss(num_classes=args.num_classes)
        self.nums = [25767, 231, 778, 243, 366, 324, 1936, 329]
        self.sum_nums = sum(self.nums)
        self.weight = args.weight
        self.mask_rate = [args.mask_rate[0]] + [args.mask_rate[1]] * (args.num_classes - 1)
        if args.multi_cls:
            self.alpha[1] = -1.0
    def forward(self, preds, labels):
        loss = 0.0
        if not self.bce:
            for i in range(len(preds)):
                alpha = 1 - self.nums[i] / self.sum_nums if self.weight else self.alpha[i]
                preds_cls = preds[i]
                labels_cls = labels[:, i]
                if i > 0 and self.multi_cls:
                    labels_cls = labels.argmax(dim=1)
                    if self.multi_pos: # 用多分类器，且用了Multi-positive损失
                        lossvalue = self.multiad_criterion(preds_cls, labels_cls)
                    else: # 用多分类器，单标签交叉熵损失
                        lossvalue = self.criterion(preds_cls, labels_cls, cls=i, alpha=alpha, gamma=self.gamma, mask_rate=self.mask_rate[i])
                else: # 没用多分类器
                    lossvalue = self.criterion(preds_cls, labels_cls, cls=i, alpha=alpha, gamma=self.gamma, mask_rate=self.mask_rate[i])
                # print("ce" + str(i), lossvalue)
                loss += lossvalue / len(preds)
        else:
            for i in range(len(preds)):
                preds_cls = preds[i]
                labels_cls = labels[:, i]
                if i > 0 and self.multi_cls:
                    if self.multi_pos: # 用多分类器，且用了Multi-positive损失
                        lossvalue = self.multiad_criterion(preds_cls, labels)
                    elif self.bce: # 用多分类器，多标签交叉熵损失，没用multi-positive损失
                        labels_cls = labels[:,:self.num_classes]
                        lossvalue = self.bce_criterion(preds_cls, labels_cls.float())
                    else: # 用多分类器，单标签交叉熵损失
                        lossvalue = self.bce_criterion(preds_cls, labels)
                else: # 没用多分类器
                    lossvalue = self.criterion(preds_cls, labels_cls, alpha=self.alpha[i], gamma=self.gamma)

                # print("ce" + str(i), lossvalue)
                loss += lossvalue / len(preds)
    
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.有标签平滑正则化的交叉熵loss
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, epsilon=0.1, num_classes=8, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, cls, alpha=0.5, gamma=0.0, mask_rate=0.0):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        targets_c = targets.clone()
        num_classes = inputs.size()[1]
        log_probs = self.logsoftmax(inputs)
        probs = self.softmax(inputs)
        targets_c = torch.zeros(log_probs.size()).scatter_(1, targets_c.unsqueeze(1).data.cpu(), 1).cuda()
        # printt = True if alpha == 0.75 and targets[:,0].sum() < targets.size()[0] else False
        printt = False
        probs_t = probs * targets_c + (1 - probs) * (1 - targets_c)
        targets_c = (1 - self.epsilon) * targets_c + self.epsilon / num_classes
        loss = - targets_c * log_probs
        if printt:
            print('loss w/o fl', loss)
        loss = loss * ((1 - probs_t) ** gamma)
        if alpha > 0:
            alpha_t = 2 * (alpha * targets_c + (1 - alpha) * (1 - targets_c)).cuda()
            loss = loss * alpha_t
        if printt:
            print('loss w fl', loss)
        
        if mask_rate > 0.0:
            pos_index = torch.nonzero(targets) if cls == 0 else torch.nonzero(1 - targets)
            neg_index = torch.nonzero(1 - targets) if cls == 0 else torch.nonzero(targets)
            # print(cls, pos_index.size(), neg_index.size())
            sel = random.sample(range(len(pos_index)), int(len(pos_index) * (1.0 - mask_rate)))
            pos_index = torch.cat((pos_index[sel], neg_index), dim=0).squeeze(1)
            loss = loss[pos_index]

        loss = loss.mean(0).sum()
        return loss


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)#把每个特征得到二范数
        dist = dist + dist.t()#两个特征二范数之和
        dist.addmm_(1, -2, inputs, inputs.t())#计算1*dist-2*inputs*inputs.t()，假设俩向量分别是a，b->a^2+b^2-2ab
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability#求开方，也就是求距离
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())#相等为1不等为0
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))#所有的正例中距离最大的
            if not mask[i].all():
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))#所有的负例中距离最小的
            else:
                dist_an.append(torch.tensor([1.0]).cuda())
        dist_ap = torch.cat(dist_ap)#anchor和正例的距离
        dist_an = torch.cat(dist_an)#anchor和负例的距离
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)#生成和dist_an一样大小的全一矩阵
        loss = self.ranking_loss(dist_an, dist_ap, y)#=max(0,-y*(x1-x2)+margin)也就是最后想要x1-x2越来越大
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class MultiPositiveLoss(nn.Module):
    def __init__(self, num_classes):
        super(MultiPositiveLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets_c = targets.clone()
        # print("target", targets_c)
        mask = torch.zeros(inputs.size()).scatter_(1, targets_c.unsqueeze(1).data.cpu(), 1).cuda()
        # print("mask", mask)
        if_zero = targets_c.nonzero()
        positive_mask = torch.zeros(inputs.size(), dtype=bool).cuda()
        for cls in range(1, inputs.size()[1]):
            targets_c[if_zero] = cls
            positive_mask |= torch.zeros(inputs.size(), dtype=bool).scatter_(1, targets_c.unsqueeze(1).data.cpu(), 1).cuda()

        negative_mask = ~positive_mask
        # print("negative mask", negative_mask)
        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negative_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        loss = (- mask * log_prob).sum(1).mean()
        return loss


class LogicTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0):
        super(LogicTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        #先寻找targets为0/1的
        #inputs = torch.tensor([item.cpu().detach().numpy() for item in inputs]).cuda()
        #inputs = torch.tensor(inputs).cuda()
        log_probs = self.softmax(inputs)
        #print(temp)
        targets = targets.cuda()
        zero = torch.zeros(targets.size()).cuda()
        one = torch.ones(targets.size()).cuda()
        if_zero = targets.eq(zero).cuda()#有问题
        if_one = targets.eq(one).cuda()#正常

        #indexZero = torch.tensor(np.nonzero(if_zero)).cuda()[0]
        #indexOne = torch.tensor(np.nonzero(if_one)).cuda()[0]
        if if_zero.all():#如果全有问题
            #print ('here1')
            indexZero = torch.tensor(np.nonzero(if_zero)).cuda()
            rightOut = torch.cat([log_probs[indexZero][1]]).cuda()
        elif if_one.all():#如果全没问题
            #print ('here2')
            indexOne = torch.tensor(np.nonzero(if_one)).cuda()
            rightOut = torch.cat([log_probs[indexOne][1]]).cuda()
        else:
            #print ('here3')
            indexZero = torch.tensor(np.nonzero(if_zero)).cuda()
            indexOne = torch.tensor(np.nonzero(if_one)).cuda()
            #print('input:',inputs.size(),inputs)
            if indexZero.size()[0]==1:
                indexZero
            rightOut = torch.cat([log_probs[indexZero][0], log_probs[indexOne][1]]).cuda()

        myMargin = (torch.ones(rightOut.size()) * 0.7).cuda()
        y = torch.ones(rightOut.size()).cuda()  # 生成和dist_an一样大小的全一矩阵
        loss = self.ranking_loss(rightOut, myMargin, y).cuda()
        #print('loss:',loss,'rightOut:',rightOut,'myMargin:',myMargin)
        #0.7 - rightOut < 0 -> rightOut > 0.7
        # compute accuracy
        correct = torch.ge(rightOut, myMargin).sum().item()
        return loss,correct

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device='cuda'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #   2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #   256 x 512
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   #   256 x   512
            anchor_count = contrast_count   #   2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        #   print (anchor_dot_contrast.size())  256 x 256

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


