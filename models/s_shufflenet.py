import math
import torch
import torch.nn as nn


from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear
from utils.config import FLAGS


class ShuffleModule(nn.Module):
    def __init__(self, groups):
        super(ShuffleModule, self).__init__()
        self.groups = groups
        self.ignore_model_profiling = True

    def forward(self, x):
        b, n, h, w = x.size()
        x = x.view(b, self.groups, n//self.groups, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


class Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        # residual or concat
        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            block_outp = [i - j for i, j in zip(outp, inp)]
            self.concat_branch = nn.AvgPool2d(3, 2, 1)
        else:
            block_outp = outp

        # first group
        if stride == 2 and max(inp) == (24 * max(FLAGS.width_mult_list)):
            self.first_group = 1
        else:
            self.first_group = FLAGS.groups

        inp_split = [i//self.first_group for i in inp]
        midp = [i//FLAGS.width_compress for i in outp]
        lastp = [i//FLAGS.groups for i in block_outp]
        firstp = [i//FLAGS.groups for i in midp]
        self.firstp = firstp
        self.inp = inp
        self.midp = midp
        self.lastp = lastp
        self.width_mult = max(FLAGS.width_mult_list)
        if self.first_group == 1:
            layers_a = [
                nn.Sequential(
                    SlimmableConv2d(inp_split, firstp, 1, 1, 0, bias=False),
                    SwitchableBatchNorm2d(firstp),
                    nn.ReLU(inplace=True),
                )
                for _ in range(FLAGS.groups)
            ]
        else:
            layers_a = [
                nn.Sequential(
                    SlimmableConv2d(inp_split, firstp, 1, 1, 0, bias=False),
                    SwitchableBatchNorm2d(firstp),
                    nn.ReLU(inplace=True),
                )
                for _ in range(FLAGS.groups)
            ]
        layers_b = [
            ShuffleModule(FLAGS.groups),
            SlimmableConv2d(
                midp, midp, 3, stride, 1, groups_list=midp, bias=False),
            SwitchableBatchNorm2d(midp),
            # nn.ReLU(inplace=True),
        ]
        midp_split = [i//FLAGS.groups for i in midp]
        lastp = [i//FLAGS.groups for i in block_outp]
        layers_c = [
            nn.Sequential(
                SlimmableConv2d(midp_split, lastp, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(lastp),
            )
            for _ in range(FLAGS.groups)
        ]
        # NOTE: nn.ModuleList does not have a forward, thus forward hook will
        # fail. A trick is used here to support model profiling.
        self.a_len = len(layers_a)
        self.b = nn.Sequential(*layers_b)
        self.c_len = len(layers_c)
        for i in range(len(layers_a)):
            setattr(self, 'a_{}'.format(i), layers_a[i])
        for i in range(len(layers_c)):
            setattr(self, 'c_{}'.format(i), layers_c[i])
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = x
            x_split = torch.split(res, list(res.size())[1]//self.a_len, dim=1)
            res = torch.cat(
                [getattr(self, 'a_{}'.format(i))(x_split[i]) for i in range(
                    self.a_len)], 1)
            res = self.b(res)
            x_split = torch.split(res, list(res.size())[1]//self.c_len, dim=1)
            res = torch.cat(
                [getattr(self, 'c_{}'.format(i))(x_split[i]) for i in range(
                    self.c_len)], 1)
            res += x
            res = self.post_relu(res)
        else:
            res = x
            concat = self.concat_branch(x)
            if self.first_group == 1:
                res = torch.cat(
                    [getattr(self, 'a_{}'.format(i))(x) for i in range(
                        self.a_len)], 1)
            else:
                x_split = torch.split(
                    res, list(res.size())[1]//self.a_len, dim=1)
                res = torch.cat(
                    [getattr(self, 'a_{}'.format(i))(
                        x_split[i]) for i in range(self.a_len)], 1)
            res = self.b(res)
            x_split = torch.split(res, list(res.size())[1]//self.c_len, dim=1)
            res = torch.cat(
                [getattr(self, 'c_{}'.format(i))(x_split[i]) for i in range(
                    self.c_len)], 1)
            res = self.post_relu(res)
            res = torch.cat([res, concat], 1)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        self.block_setting = [
            # c, s
            # stage 2
            [240, 2],
            [240, 1],
            [240, 1],
            [240, 1],
            # stage 3
            [480, 2],
            [480, 1],
            [480, 1],
            [480, 1],
            [480, 1],
            [480, 1],
            [480, 1],
            [480, 1],
            # stage 4
            [960, 2],
            [960, 1],
            [960, 1],
            [960, 1],
        ]

        self.features = []

        channels = [int(24*width_mult) for width_mult in FLAGS.width_mult_list]
        first_stride = 2
        group_channels = [i//FLAGS.groups for i in channels]
        head = [
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], group_channels, 3,
                    first_stride, 1, bias=False),
                SwitchableBatchNorm2d(group_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            for _ in range(FLAGS.groups)
        ]
        for i in range(len(head)):
            setattr(self, 'head_{}'.format(i), head[i])

        for c, s in self.block_setting:
            outp = [int(c*width_mult) for width_mult in FLAGS.width_mult_list]
            self.features.append(Block(channels, outp, s))
            channels = outp

        avg_pool_size = input_size//32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(
          SlimmableLinear(
            channels,
            [num_classes for _ in range(len(channels))])
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = torch.cat(
            [getattr(self, 'head_{}'.format(i))(x) for i in range(
                FLAGS.groups)], 1)
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
