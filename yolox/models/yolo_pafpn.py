#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .attention import CBAM, SE, ECA, CoordAtt
from .ASFF import ASFF

class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # self.cbam_1 = CBAM(int(in_channels[2] * width))  # 对应dark5输出的1024维度通道
        # self.cbam_2 = CBAM(int(in_channels[1] * width))  # 对应dark4输出的512维度通道
        # self.cbam_3 = CBAM(int(in_channels[0] * width))  # 对应dark3输出的256维度通道
        # 使用时，注释上面或者下面的代码
        # self.eca_1 = ECA(int(in_channels[2] * width))  # 对应dark5输出的1024维度通道
        # self.eca_2 = ECA(int(in_channels[1] * width))  # 对应dark4输出的512维度通道
        # self.eca_3 = ECA(int(in_channels[0] * width))  # 对应dark3输出的256维度通道

        # self.SE_1 = SE(int(in_channels[2] * width))  # 对应dark5输出的1024维度通道
        # self.SE_2 = SE(int(in_channels[1] * width))  # 对应dark4输出的512维度通道
        # self.SE_3 = SE(int(in_channels[0] * width))  # 对应dark3输出的256维度通道

        # self.ca_1 = CoordAtt(inp=int(in_channels[2] * width), oup=int(in_channels[2] * width))
        # self.ca_2 = CoordAtt(inp=int(in_channels[1] * width), oup=int(in_channels[1] * width))
        # self.ca_3 = CoordAtt(inp=int(in_channels[0] * width), oup=int(in_channels[0] * width))
        #
        # self.asff_1 = ASFF(level=0, multiplier=width)
        # self.asff_2 = ASFF(level=1, multiplier=width)
        # self.asff_3 = ASFF(level=2, multiplier=width)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        # 3、直接对输入的特征图使用注意力机制
        # x0 = self.cbam_1(x0)
        # x1 = self.cbam_2(x1)
        # x2 = self.cbam_3(x2)
        # 使用时，注释上面或者下面的代码
        # x0 = self.eca_1(x0)
        # x1 = self.eca_2(x1)
        # x2 = self.eca_3(x2)

        # x0 = self.SE_1(x0)
        # x1 = self.SE_2(x1)
        # x2 = self.SE_3(x2)

        # x0 = self.ca_1(x0)
        # x1 = self.ca_2(x1)
        # x2 = self.ca_3(x2)


        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16

        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32

        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        #asff
        # pan_out0 = self.asff_1(outputs)
        # pan_out1 = self.asff_2(outputs)
        # pan_out2 = self.asff_3(outputs)
        # outputs = (pan_out2, pan_out1, pan_out0)

        return outputs
