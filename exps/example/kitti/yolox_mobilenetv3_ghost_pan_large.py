#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.width = 0.5
        self.input_size = (256, 832)  # (height, width)
        self.test_size = (256, 832)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_num_workers = 8
        self.max_epoch = 300
        self.mosaic_prob = 0
        self.num_classes = 7
        self.train_ann = "train.json"
        # name of annotation file for evaluation
        self.val_ann = "val.json"
        # name of annotation file for testing
        self.test_ann = "test.json"
        self.data_dir = "E:\Datacrawling\YOLOX\KITTI_YOLOX\img"

        self.no_aug_epochs = 80
        self.eval_interval = 10

    def get_model(self):
        from yolox.models import YOLOGhostPAFPNMobileNetv3

        from yolox.models import YOLOX, YOLOXHeadFixed

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels_fpn = [40, 112, 960]
            in_channels_neck = [96,96,96]
            backbone = YOLOGhostPAFPNMobileNetv3(out_indices=(6, 12, 16),in_channels=in_channels_fpn,)
            head = YOLOXHeadFixed(self.num_classes, width=self.width, in_channels=in_channels_neck, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model