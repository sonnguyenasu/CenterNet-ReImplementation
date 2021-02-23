from torchvision.models import resnet50
import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self, resnet_model=resnet50(pretrained=True)):
        super(ResNet,self).__init__()
        self.layer1 = nn.Sequential(*list(resnet_model.children())[:4])
        self.layer2 = resnet_model.layer1
        self.layer3 = resnet_model.layer2
        self.layer4 = resnet_model.layer3
        self.layer5 = resnet_model.layer4
    def forward(self,x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return c1, c2, c3, c4, c5


if __name__ == '__main__':
    rn = ResNet()
    import torch
    x = torch.rand((1,3,800,1024))
    y = rn(x)
    for i in range(5):
        print(y[i].shape)