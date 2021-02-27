from torch import nn
from .resnet import ResNet
from .fpn import FPN
import torch


class CenterNet(nn.Module):
    def __init__(self, backbone='resnet50', start_at=256, num_class=6):
        super(CenterNet, self).__init__()
        blocks = [start_at, start_at*2, start_at*4, start_at*8]
        self.num_class = num_class
        if backbone == 'resnet50':
            self.backbone = ResNet()
        self.fpn = FPN(in_blocks=blocks)
        self.prediction = nn.Conv2d(256, num_class+4, 1)
        self.center_feed = nn.Conv2d(num_class,num_class,1)
        self.size_feed = nn.Conv2d(2,2,1)
        self.center_feed.bias.data.fill_(-2.19)
        self.size_feed.bias.data.fill_(5.8)
        # self.offset_feed = nn.Conv2d(2, 2, 1)
        # self.size_feed = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        features = self.backbone(x)[1:]
        feed = self.fpn(features)
        prediction = self.prediction(feed)
        center_predict, offset_predict, size_predict = torch.split(
            prediction, [self.num_class, 2, 2], 1)
        center_predict = torch.sigmoid(self.center_feed(center_predict))
        offset_predict = torch.exp(offset_predict)
        size_predict = torch.exp(self.size_feed(size_predict))
        prediction = torch.cat(
            [center_predict, offset_predict, size_predict], dim=1)
        return prediction


if __name__ == '__main__':
    import torch
    x = torch.rand((1, 3, 800, 1024))
    cnet = CenterNet()
    y = cnet(x)
    print(y.shape)
