import torch.nn as nn
import math


from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear
from .slimmable_ops import pop_channels
from utils.config import FLAGS


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, stride):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]

        layers = [
            SlimmableConv2d(
                inp, inp, 3, stride, 1, groups_list=inp, bias=False),
            SwitchableBatchNorm2d(inp),
            nn.ReLU6(inplace=True),

            SlimmableConv2d(inp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
            nn.ReLU6(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        channel_num_list = FLAGS.channel_num_list.copy()

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = pop_channels(FLAGS.channel_num_list)
        first_stride = 2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 3,
                    first_stride, 1, bias=False),
                SwitchableBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for c, n, s in self.block_setting:
            for i in range(n):
                outp = pop_channels(FLAGS.channel_num_list)
                if i == 0:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, s))
                else:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, 1))
                channels = outp

        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        self.outp = channels
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        FLAGS.channel_num_list = channel_num_list.copy()

        # classifier
        self.classifier = nn.Sequential(
            SlimmableLinear(
                self.outp,
                [num_classes for _ in range(len(self.outp))]
            )
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
