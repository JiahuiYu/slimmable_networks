import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint as official_load_checkpoint


import mmdet.flags as FLAGS


class SwitchableBatchNorm2d(nn.Module):
     def __init__(self, num_features_list):
         super(SwitchableBatchNorm2d, self).__init__()
         self.num_features_list = num_features_list
         self.num_features = max(self.num_features_list)
         self.width_mult = FLAGS.width_mult #  FLAGS.width_max_mult
         self.onehot = torch.zeros(1, self.num_features, 1, 1)
         self.ignore_model_profiling = True
         self.bn = nn.ModuleList(
             [nn.BatchNorm2d(i) for i in num_features_list])

     def forward(self, input):
         y = self.bn[FLAGS.width_mult_list.index(self.width_mult)](input)
         return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.current_in_channels = -1
        self.current_out_channels = -1
        self.current_groups = -1
        self.width_mult = FLAGS.width_mult
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight[
            0:self.current_out_channels, 0:self.current_in_channels, :, :]
        if self.bias:
            bias = self.bias[0:self.current_out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.current_groups)
        return y


def load_checkpoint(m, pretrained, strict=False, logger=None):
    """Docstring for load_checkpoint"""
    checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
    # update keys from external models
    if type(checkpoint) == dict and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    new_checkpoint = {}
    new_keys = list(m.state_dict().keys())
    keys = []
    for k in new_keys:
        if 'num_batches_tracked' in k:
            continue
        keys.append(k)
    new_keys = keys
    old_keys = list(checkpoint.keys())
    for key_new, key_old in zip(new_keys, old_keys):
        new_checkpoint[key_new] = checkpoint[key_old]
        # print('remap {} to {}'.format(key_new, key_old))
    checkpoint = new_checkpoint
    m.load_state_dict(checkpoint)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return SlimmableConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False):
        """Bottleneck block.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        if style == 'pytorch':
            conv1_stride = 1
            conv2_stride = stride
        else:
            conv1_stride = stride
            conv2_stride = 1
        self.conv1 = SlimmableConv2d(
            inplanes, planes, kernel_size=1, stride=conv1_stride, bias=False)
        self.bn1 = SwitchableBatchNorm2d(planes)
        self.conv2 = SlimmableConv2d(
            planes,
            planes,
            kernel_size=3,
            stride=conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.bn2 = SwitchableBatchNorm2d(planes)
        self.conv3 = SlimmableConv2d(
            planes, [i * self.expansion for i in planes], kernel_size=1, bias=False)
        self.bn3 = SwitchableBatchNorm2d([i * self.expansion for i in planes])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            SlimmableConv2d(
                inplanes,
                [i * block.expansion for i in planes],
                kernel_size=1,
                stride=stride,
                bias=False),
            SwitchableBatchNorm2d([i * block.expansion for i in planes]),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp))
    inplanes = [i * block.expansion for i in planes]
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 with_cp=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        assert num_stages >= 1 and num_stages <= 4
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_cp = with_cp

        self.inplanes = [int(i*64) for i in FLAGS.width_mult_list]
        self.conv1 = SlimmableConv2d(
            [3 for _ in range(len(FLAGS.width_mult_list))], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = SwitchableBatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = [int(j * 64 * 2**i) for j in FLAGS.width_mult_list]
            res_layer = make_res_layer(
                block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp)
            self.inplanes = [int(j * block.expansion) for j in planes]
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = block.expansion * 64 * 2**(len(stage_blocks) - 1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, FLAGS.pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, SlimmableConv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                if FLAGS.width_mult_current < 1.0:
                    bs = x.size()
                    y = torch.cat([x, torch.cuda.FloatTensor(bs[0], int((1-FLAGS.width_mult_current) * (bs[1]/FLAGS.width_mult_current)), bs[2], bs[3]).fill_(0)], 1)
                else:
                    y = x
                outs.append(y)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, SwitchableBatchNorm2d):
                    m.eval()
                    for params in m.parameters():
                        params.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            # slimmable
            for bn in self.bn1.bn:
                for param in bn.parameters():
                    param.requires_grad = False
                bn.eval()
                bn.weight.requires_grad = False
                bn.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

            self.apply(lambda m: freeze_params(m) if isinstance(m, SwitchableBatchNorm2d) else None)


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
        # print(p.size())
