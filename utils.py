import xml.etree.ElementTree as ET
from lxml.etree import parse
import os
from PIL import Image, ImageDraw,ImageFont

def read_xml(xml_path):
    '''
    读取xml模板
    args:
        xml_path: xml文件路径
    '''
    try:
        tree = parse(xml_path)
        root = tree.getroot()
        return tree,root
    except:
        return None
    
def make_dirs(path):
    '''
    目录不存在就创建
    args:
        path: 要创建的目录
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    
    color_map = [(0,230,0),(255,255,0),(0,255,255),(255,0,255),(255,0,0)]
    return color_map

def get_color_map_list_(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map

def draw_box(im, np_boxes, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of box
    Returns:
        im (PIL.Image.Image): visualized image
    """
    draw_thickness = min(im.size) // 300
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    font_path = "F:/Datasets/matting/Image_Text/font/微软雅黑.ttf"
    font_size = 15
    font = ImageFont.truetype(font_path, font_size)

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color[clsid])

        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            # print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
            #       'right_bottom:[{:.2f},{:.2f}]'.format(
            #           int(clsid), score, xmin, ymin, xmax, ymax))
            # draw bbox
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=draw_thickness,
                fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)

        # draw label
        # text = "{} {:.4f}".format(labels[clsid], score)
        text = "{}".format(labels[clsid])
        tw, th = draw.textsize(text)
        tw = len(text)*int(font_size*2/3)
        th = font_size
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        # draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
        draw.text((xmin + 1, ymin - th), text, fill=(0, 0, 0),font=font)
    return im

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