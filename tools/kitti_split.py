'''
用于将KITTI数据集的7000多张训练集分为：前4000张为训练集，4000-6000张为验证集，剩余为测试集
运行命令：
python ./tools/kitti_split.py --source_img_path ./KITTI_origin/training/image_2 --source_label_path ./KITTI_origin/training/label_2/ --dst_img_path ./KITTI_YOLOX/img --dst_label_path ./KITTI_YOLOX/label
'''


import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("")   
    parser.add_argument('--source_img_path', help="Specify original kitti img path")
    parser.add_argument('--source_label_path', help="Specify original kitti label path")

    parser.add_argument('--dst_img_path', help="Specify splited kitti img path")
    parser.add_argument('--dst_label_path', help="Specify splited kitti label path")

    return parser

def check_dir(dir):
    if Path(dir).is_dir() == False:
        Path(dir).mkdir(parents=True, exist_ok=True)
        logger.info('Created %s' % dir)

if __name__ == '__main__':
    args = make_parser().parse_args()

    img_root = Path(args.source_img_path)
    label_root = Path(args.source_label_path)

    img_list = os.listdir(img_root)
    dst_train_img_root = Path(args.dst_img_path)/'train'
    dst_val_img_root = Path(args.dst_img_path)/'val'
    dst_test_img_root = Path(args.dst_img_path)/'test'

    dst_train_label_root = Path(args.dst_label_path)/'train'
    dst_val_label_root = Path(args.dst_label_path)/'val'
    dst_test_label_root = Path(args.dst_label_path)/'test'

    check_dir(dst_train_img_root)
    check_dir(dst_val_img_root)
    check_dir(dst_test_img_root)
    check_dir(dst_train_label_root)
    check_dir(dst_val_label_root)
    check_dir(dst_test_label_root)

    for img_name in tqdm(img_list):
        if int(Path(img_name).stem) < 4000:
            shutil.copyfile(img_root/img_name, dst_train_img_root/img_name)
            shutil.copyfile(label_root/(Path(img_name).with_suffix('.txt')), dst_train_label_root/(Path(img_name).with_suffix('.txt')))

        elif int(Path(img_name).stem) < 6000:
            shutil.copyfile(img_root/img_name, dst_val_img_root/img_name)
            shutil.copyfile(label_root/(Path(img_name).with_suffix('.txt')), dst_val_label_root/(Path(img_name).with_suffix('.txt')))

        else:
            shutil.copyfile(img_root/img_name, dst_test_img_root/img_name)
            shutil.copyfile(label_root/(Path(img_name).with_suffix('.txt')), dst_test_label_root/(Path(img_name).with_suffix('.txt')))

