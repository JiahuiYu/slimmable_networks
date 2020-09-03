import torch.nn as nn
import math


from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear
from .slimmable_ops import pop_channels
from utils.config import FLAGS


class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, cmid, kernel_size):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        if cmid != inp:
            layers += [
                SlimmableConv2d(inp, cmid, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(cmid),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SlimmableConv2d(
                cmid, cmid, kernel_size, stride,
                kernel_size // 2, groups_list=cmid, bias=False),
            SwitchableBatchNorm2d(cmid),
            nn.ReLU6(inplace=True),
            SlimmableConv2d(cmid, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s, k
            [1, 16, 1, 1, 3],
            [3, 24, 1, 2, 3],
            [3, 24, 1, 1, 3],
            [3, 24, 1, 1, 3],
            [3, 40, 1, 2, 5],
            [3, 40, 1, 1, 5],
            [3, 40, 1, 1, 5],
            [6, 80, 1, 2, 5],
            [6, 80, 1, 1, 5],
            [6, 80, 1, 1, 5],
            [6, 96, 1, 1, 3],
            [6, 96, 1, 1, 3],
            [6, 192, 1, 2, 5],
            [6, 192, 1, 1, 5],
            [6, 192, 1, 1, 5],
            [6, 192, 1, 1, 5],
            [6, 320, 1, 1, 3],
        ]

        channel_num_list = FLAGS.channel_num_list.copy()

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = pop_channels(FLAGS.channel_num_list)
        first_stride = 1 if FLAGS.dataset == 'cifar10' else 2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 3,
                    first_stride, 1, bias=False),
                SwitchableBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for index, (t, c, n, s, k) in enumerate(self.block_setting):
            if index == 0:
                outp = pop_channels(FLAGS.channel_num_list)
                cmid = channels
            else:
                cmid = pop_channels(FLAGS.channel_num_list)
                outp = pop_channels(FLAGS.channel_num_list)
            self.features.append(InvertedResidual(channels, outp, s, cmid, k))
            channels = outp

        # tail
        self.outp = pop_channels(FLAGS.channel_num_list)
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(channels, self.outp, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(self.outp),
                nn.ReLU6(inplace=True)
            )
        )
        # cifar10
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        FLAGS.channel_num_list = channel_num_list.copy()

        # classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(p=FLAGS.dropout_rate),
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
