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
        self.input_size = (768, 1280)  # (height, width)
        self.test_size = (768, 1280)
        # self.input_size = (256, 832)  # (height, width)
        # self.test_size = (256, 832)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # self.enable_mixup = False
        self.data_num_workers = 4
        self.max_epoch = 100
        self.num_classes = 5
        self.data_dir = "E:\Datacrawling\YOLOX\datasets\COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.no_aug_epochs = 100
        self.eval_interval = 1
        self.enable_mixup = False

    def get_model(self):
        from yolox.models import YOLOPAFPNMobilenetv3

        from yolox.models import YOLOX, YOLOXHeadFixed

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [40, 112, 960]
            backbone = YOLOPAFPNMobilenetv3(out_indices=(6, 12, 16), in_channels=in_channels, )
            head = YOLOXHeadFixed(self.num_classes, width=self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model