'''
KITTI标注转COCO标注

运行命令：

（1）训练集：python tools/kitti2coco.py --img_path ./KITTI_YOLOX/img/train --label_path ./KITTI_YOLOX/label/train --dst_json ./train.json
（2）验证集：python tools/kitti2coco.py --img_path ./KITTI_YOLOX/img/val --label_path ./KITTI_YOLOX/label/val --dst_json ./val.json
（3）测试集：python tools/kitti2coco.py --img_path ./KITTI_YOLOX/img/test --label_path ./KITTI_YOLOX/label/test --dst_json ./test.json
'''
import os
import json
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

def make_parser():
    parser = argparse.ArgumentParser("Kitti to COCO format")
    parser.add_argument('--img_path', type=str, default=None, help='Specify img path')
    parser.add_argument('--label_path', type=str, default=None, help='Specify label path')
    parser.add_argument('--dst_json', type=str, default=None, help='Specify generated json file name')

    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    img_root = Path(args.img_path)
    label_root = Path(args.label_path)

    category_dict = {
        1:'Car', 
        2:'Van', 
        3:'Pedestrian', 
        4:'Person_sitting', 
        5:'Truck',  
        6:'Cyclist', 
        7:'Tram'
    }

    category_name2id_dict = {v:k for k, v in category_dict.items()}


    img_list = os.listdir(img_root)

    img_id = 0
    anno_id = 0

    json_images_list = list()
    json_annotations_list = list()
    json_categories_list = list()

    for img_name in tqdm(img_list):

        img = cv2.imread(str(img_root/img_name))
        img_height, img_width, _ = img.shape
        img_dict = {
            'license': None,
            'file_name': img_name,
            'coco_url': None,
            'height': img_height, 
            'width': img_width, 
            'date_captured': None, 
            'flickr_url': None,
            'id': img_id
        }
        json_images_list.append(img_dict)
        
        label_name = Path(img_name).with_suffix('.txt')
        with open(label_root/label_name) as f:
            anno_list = [x.split() for x in f.read().strip().splitlines()]
        for anno in anno_list:
            if anno[0] in category_name2id_dict:
                bbox = [float(anno[4]), float(anno[5]), 
                        float(anno[6])-float(anno[4]), float(anno[7])-float(anno[5])] #   anno[4:8]
                area = bbox[2]*bbox[3]
                
                anno_dict = {
                    'segmentation': None,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': img_id,
                    'bbox': bbox, 
                    'category_id': category_name2id_dict[anno[0]],
                    'id': anno_id
                }
                json_annotations_list.append(anno_dict)
                anno_id += 1
        img_id += 1
        
    for id in category_dict:
        json_categories_list.append({
            'supercategory': None,
            'id': id,
            'name': category_dict[id]
        })
        
    json_dict = {
        'images': json_images_list,
        'annotations': json_annotations_list,
        'categories': json_categories_list
    }


    with open(args.dst_json,"w") as f:
        json.dump(json_dict,f)






    
    
    
    
    
    

