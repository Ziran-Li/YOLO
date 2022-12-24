#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50


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
        #self.enable_mixup = False