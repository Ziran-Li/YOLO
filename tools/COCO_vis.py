'''
验证转换后的json格式标注的准确性。
运行命令：python tools/COCO_vis.py --img_root ./KITTI_YOLOX/img/train --label_file ./KITTI_YOLOX/train.json
'''

import argparse
from pathlib import Path
import numpy as np
import cv2
from pycocotools.coco import COCO


def make_parser():    
    parser = argparse.ArgumentParser("")        
    parser.add_argument('--img_root', type=str, default=None, help='Specify img path')        
    parser.add_argument('--label_file', type=str, default=None, help='Specify COCO format label file')        
    return parser  

if __name__ == '__main__':
    args = make_parser().parse_args()
        
    img_root = args.img_root
    anno_file = args.label_file

    coco = COCO(anno_file)
    img_ids = coco.getImgIds()

    category_list = coco.loadCats(coco.getCatIds())
    label_id2name = dict([(item['id'], item['name']) for item in category_list])

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        print('img name: ', str(Path(img_root)/img_info['file_name']))
        img = cv2.imread(str(Path(img_root)/img_info['file_name']))
        
        img_width = img_info["width"]
        img_height = img_info["height"]
        anno_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        result_anno_list = list()
        
        for anno_id in anno_ids:
            annotation = coco.loadAnns(anno_id)
            x1 = np.max((0, annotation[0]["bbox"][0]))
            y1 = np.max((0, annotation[0]["bbox"][1]))
            x2 = np.min((img_width, x1 + np.max((0, annotation[0]["bbox"][2]))))
            y2 = np.min((img_height, y1 + np.max((0, annotation[0]["bbox"][3]))))
            
            
            label = label_id2name[annotation[0]['category_id']]
            result_anno_list.append([label, x1, y1, x2, y2])

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
            cv2.putText(img, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,255,255))
        cv2.imshow('img', img)
        ret = cv2.waitKey(0)    
        if ret == 27:
            exit(0)