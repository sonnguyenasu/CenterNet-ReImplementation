import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self, l_focal=1.0, l_size=0.1, l_offset=1.0, alpha=2, beta=4):
        super(Loss, self).__init__()
        self.l_size = l_size
        self.l_offset = l_offset
        self.alpha = alpha
        self.beta = beta
        self.l_focal = l_focal

    def size_loss(self, pred, target):
        loss = 0
        pos_ids = 1-target.lt(1).float()
        pred = pred*pos_ids#[pos_ids == 0] = 0
        #pred /= (target + 1e-4)
        loss = torch.nn.functional.smooth_l1_loss(
            pred, target, size_average=False)
        num_pos = int(pos_ids.sum()/2)
        loss /= (num_pos+1e-4)
        # print(loss)
        return loss

    def offset_loss(self, pred, target, pos_ids):
        loss = 0
        #pos_ids = 1-target.lt(1).float()
        pred = pred*pos_ids# == 0] = 0

        loss = torch.nn.functional.smooth_l1_loss(
            pred, target, size_average=False)
        num_pos = int(pos_ids.sum()/2)
        loss /= (num_pos+1e-4)
        # print(loss)
        return loss

    def focal_loss(self, pred, target):
        loss = 0
        neg_ids = target.lt(1).float()
        pos_ids = 1-neg_ids
        pred = torch.where(pred > 0.99, torch.tensor(0.99,device='cuda'),pred)
        pred = torch.where(pred < 0.01, torch.tensor(0.01,device='cuda'),pred)
        neg_weights = torch.pow(1-target, self.beta)
        pos_loss = torch.log(pred)*torch.pow(1-pred, self.alpha)*pos_ids
        neg_loss = torch.log(1-pred)*torch.pow(pred,
                                               self.alpha)*neg_weights*neg_ids
        num_pos = pos_ids.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # print(num_pos)
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def forward(self, prediction, target):
        center_predict, offset_predict, size_predict = prediction
        center_mask, offset_mask, size_mask = target
        pos_ids = 1-size_mask.lt(1).float()
        focal_loss = self.focal_loss(center_predict, center_mask)
        size_loss = self.size_loss(size_predict, size_mask)
        offset_loss = self.offset_loss(offset_predict, offset_mask, pos_ids)
        total_loss = self.l_focal*focal_loss + self.l_size * \
            size_loss + self.l_offset*offset_loss
        return total_loss, size_loss, offset_loss, focal_loss
