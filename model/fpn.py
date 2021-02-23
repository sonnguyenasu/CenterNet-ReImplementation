from torch import nn
from .utils import conv_transition, conv_block, up_conv


class FPN(nn.Module):
    def __init__(self, in_blocks=(256, 512, 1024, 2048)):
        super(FPN, self).__init__()
        self.trans_2 = conv_transition(in_blocks[0], 256)
        self.trans_3 = conv_transition(in_blocks[1], 256)
        self.trans_4 = conv_transition(in_blocks[2], 256)
        self.trans_5 = conv_transition(in_blocks[3], 256)
        self.up_4 = up_conv(256, 256)
        self.up_3 = up_conv(256, 256)
        self.up_2 = up_conv(256, 256)
        self.smooth_2 = conv_transition(256, 256)
        #self.smooth_4 = conv_transition(256, 256)
        #self.smooth_3 = conv_transition(256, 256)
        #self.down_6 = up_conv(256, 256, up_scale=1/2)
        #self.down_7 = up_conv(256, 256, up_scale=1/2)

    def forward(self, x):
        c2, c3, c4, c5 = x
        p5 = self.trans_5(c5)
        p4 = self.up_4(p5)
        p3 = self.up_3(p4)
        p2 = self.up_2(p3)
        p2 = self.smooth_2(p2)
        #p3 = self.smooth_3(p3)
        #p4 = self.smooth_4(p4)
        #p6 = self.down_6(p5)
        #p7 = self.down_7(p6)
        result = p2  # [p3, p4, p5, p6, p7]
        return result


if __name__ == '__main__':
    import torch
    fpn = FPN()
    x = [torch.rand((1, 256, 200, 256)), torch.rand((1, 512, 100, 128)), torch.rand(
        (1, 1024, 50, 64)), torch.rand((1, 2048, 25, 32))]
    y = fpn(x)
    print(y.shape)
