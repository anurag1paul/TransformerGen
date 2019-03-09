from torch import nn


class ResnetBlock(nn.Module):

    def __init__(self, dim, use_dropout=True):
        super(ResnetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                bias=nn.InstanceNorm2d),
                      nn.BatchNorm2d(dim),
                      nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                 bias=nn.InstanceNorm2d),
                       nn.BatchNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

