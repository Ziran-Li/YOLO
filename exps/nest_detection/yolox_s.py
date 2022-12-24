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
        # self.input_size = (832, 1280)
        # self.mosaic_scale = (0.5, 1.5)
        # self.random_size = (10, 20)
        # self.test_size = (832, 1280)

        self.input_size = (768, 1280)  # (height, width)
        self.test_size = (768, 1280)
        # self.input_size = (256, 832)  # (height, width)
        # self.test_size = (256, 832)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # self.enable_mixup = False
        self.data_num_workers = 4
        self.max_epoch = 200
        self.num_classes = 1
        self.data_dir = "E:\Datacrawling\YOLOX\datasets\COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.no_aug_epochs = 200
        #self.eval_interval = 1
        #self.enable_mixup = False9