"""
 * Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
 *
 * @author Danil Osokin
 *
 * Includes code from Danil Osokin on https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch under Apache license
 * Copyright (c) 2019 Danil Osokin
 *
 * Created at : 2022-11-25
"""


from torch import nn


def conv(
    in_channels,
    out_channels,
    kernel_size=3,
    padding=1,
    bn=True,
    dilation=1,
    stride=1,
    relu=True,
    bias=True,
):
    modules = [
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
    ]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(
    in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        ),
        nn.ELU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )
