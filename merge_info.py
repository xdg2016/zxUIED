import xml.etree.ElementTree as ET
import os
import random
import cv2
from tqdm import tqdm
import numpy as np
import shutil




def read_xml(xml_path):
    '''
    读取xml模板
    '''
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return tree,root
    except:
        return None


def cal_iou(boxa, boxb):
    x1, y1, w1, h1 = boxa
    x2, y2, w2, h2 = boxb
    if (x1 > x2 + w2):
        return 0
    if (y1 > y2 + h2):
        return 0
    if (x1 + w1 < x2):
        return 0
    if (y1 + h1 < y2):
        return 0
    colInt = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)

def cal_overlap(boxa, boxb):
    x1, y1, w1, h1 = boxa
    x2, y2, w2, h2 = boxb
    if (x1 > x2 + w2):
        return 0
    if (y1 > y2 + h2):
        return 0
    if (x1 + w1 < x2):
        return 0
    if (y1 + h1 < y2):
        return 0
    colInt = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1)



def extract_merge_info(info):

    cal_info = []
    for i in range(len(info)):
        
        item = info[i]
        name,xmin,ymin,xmax,ymax = item
        w = xmax-xmin
        h = ymax-ymin
        boxa = (xmin,ymin,w,h)
        sum_tree = 0
        sum_index = []
        for j in range(len(info)):
            item1 = info[j]
            name1,xmin1,ymin1,xmax1,ymax1 = item1
            w1 = xmax1-xmin1
            h1 = ymax1-ymin1
            boxb = (xmin1,ymin1,w1,h1)
            if i==j:
                continue
            else:

                iou = cal_iou(boxa,boxb)
                overlap = cal_overlap(boxa,boxb)
                if overlap>=0.97 and iou<1.0:
                    # print(iou,overlap)
                    sum_tree+=1
                    sum_index.append(j)
        
        cal_info.append((name,xmin,ymin,xmax,ymax,sum_tree,sum_index))



    merge_info = []

    ## 获取树的最大层数
    max_sum_tree = 0
    np_info = np.array(cal_info)
    max_sum_tree = np.max(np_info[:,5])
    for i in range(len(cal_info)):
        item = cal_info[i]
        info=dict()
        tree_index = []
        name,xmin,ymin,xmax,ymax,sum_tree,sum_index = item
        for j in range(len(cal_info)):
            item1 =cal_info[j]
            name1,xmin1,ymin,xmax1,ymax1,sum_tree1,sum_index1 = item1
            if(sum_tree+1==sum_tree1):
                if(i in sum_index1):
                    tree_index.append(j)
        
        merge_info.append((name,xmin,ymin,xmax,ymax,sum_tree,tree_index))

    return merge_info


def extract_info_from_xml(xml_path):
    '''
    生成分类数据集
    '''
    # 遍历每个文件夹
    info = []
    # try:
    tree,root = read_xml(xml_path)
    # except:
    #     print("error xml!")

    objs = tree.findall('object')
    img_size = tree.find('size')
    w = int(img_size.find('width').text)
    h = int(img_size.find('height').text)
    for i,obj in enumerate(objs):
        name = str(obj.find('name').text)
        
        bndbox = obj.find('bndbox')
        delta = 2 # 外扩2个像素
        xmin = max(int(float(bndbox.find('xmin').text)) - delta,0)
        ymin = max(int(float(bndbox.find('ymin').text)) - delta,0)
        xmax = min(int(float(bndbox.find('xmax').text)) + delta,w-1)
        ymax = min(int(float(bndbox.find('ymax').text)) + delta,h-1)
        info.append((name,xmin,ymin,xmax,ymax))
    merge_info = extract_merge_info(info)
    return merge_info

xml_path = "./test/xmls/13_0.xml"
merge_info = extract_info_from_xml(xml_path)

print(merge_info)


    # if sum_tree ==0:
    #     info["type_name"] = name
    #     info["loc"] = (xmin,ymin,xmax,ymax)
    




# print(info)