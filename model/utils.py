from torch import nn


class conv_block(nn.Module):
    def __init__(self, fan_in, fan_out, kernel=3, stride=1, padding=0, use_bn = True):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(fan_in, fan_out, kernel_size=kernel,stride=stride,padding=padding)
        self.activate = nn.ReLU()
        
        if use_bn:
            self.bn = nn.BatchNorm2d(fan_out)
        else:
            self.bn = None
    def forward(self,x):
        features = self.activate(self.conv(x))
        if self.bn is not None:
            features = self.bn(features)
        return features

class conv_transition(nn.Module):
    def __init__(self,fan_in,fan_out):
        super(conv_transition,self).__init__()
        self.conv1 = nn.Conv2d(fan_in,fan_out,1)
        #self.activate= nn.LeakyReLU()
    def forward(self,x):
        return self.conv1(x)

class up_conv(nn.Module):
    def __init__(self, fan_in, fan_out, up_scale=2):
        super(up_conv,self).__init__()
        self.trans = conv_block(fan_in,fan_out,1)
        
        if up_scale > 1:
            self.up = nn.Upsample(scale_factor=up_scale)
        else:
            self.up = None
            self.conv = conv_block(fan_out,fan_out, 3,int(1/up_scale),1)#nn.AvgPool2d((int(1/up_scale),int(1/up_scale)))
    def forward(self,x):
        if self.up is None:
            return self.conv(self.trans(x))
        return self.up(self.trans(x))

if __name__ == '__main__':
    import torch
    x = torch.rand((1,512,32,25))
    up = up_conv(512,1024,2)
    y = up(x)
    print(y.shape)