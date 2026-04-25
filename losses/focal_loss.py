# @title losses/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss from the paper "Focal Loss for Dense Object Detection".
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: model output, shape [N, C]
        # targets: ground truth labels, shape [N]
        
        # Tính Cross Entropy Loss cơ bản mà không reduce
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Lấy xác suất của lớp đúng (p_t)
        # F.softmax(logits, dim=1) -> [N, C]
        # .gather(1, targets.view(-1, 1)) -> [N, 1]
        # .squeeze(1) -> [N]
        pt = torch.exp(-ce_loss)
        
        # Tính Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss