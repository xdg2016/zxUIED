import os
import random
import cv2
from tqdm import tqdm
import numpy as np
import shutil
from resnet import ResNet
from picodet import PicoDet
from multiprocessing.dummy import Pool as ThreadPool
import xml.dom.minidom
from utils import *
from lxml.etree import SubElement as subElement
from lxml.etree import Element, tostring, parse
import json

def gen_train_val(data_home):
    '''
    生成训练验证数据集，data_home路径下没有子文件夹，只有imgs和xmls
    args:
        data_home: 数据集根目录
    return:
        None
    '''
    trainval_data_home = data_home
    f_train = open(trainval_data_home+"/train.txt","w",encoding="utf-8")
    f_val = open(trainval_data_home+"/val.txt","w",encoding="utf-8")
    
    imgs = [img for img in os.listdir(os.path.join(trainval_data_home,"imgs")) if os.path.splitext(img)[-1] in ['.jpg','.png']]
    for img in imgs:
        img_name = os.path.splitext(img)[0]
        img_path = os.path.join(trainval_data_home,"imgs",img)
        xml_path = os.path.join(trainval_data_home,"xmls",img_name+".xml")
        
        dst_img_path = trainval_data_home+"/train_val/imgs/"+img
        dst_xml_path = trainval_data_home+'/train_val/xmls/'+img_name+".xml"
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue
        if random.random() > 0.1:
            f_train.write(f"{dst_img_path.replace(trainval_data_home,'.')} {dst_xml_path.replace(trainval_data_home,'.')}\r")
        else:
            f_val.write(f"{dst_img_path.replace(trainval_data_home,'.')} {dst_xml_path.replace(trainval_data_home,'.')}\r")

    f_train.close()
    f_val.close()

def gen_train_val2(date,data_home,type):
    '''
    生成训练验证数据集,data_home文件夹下有多个子文件夹，每个子文件夹下有imgs和xmls
    args:
        date: 数据日期
        data_home: 数据根目录
        type: 检测类型，block还是元素
    return:
        None
    '''
    # data_home = "F:/Datasets/UIED"
    origin_data_home = data_home+f"/裁剪标注数据/{date}"
    if type != "":
        trainval_data_home = data_home+f"/trainval_data/UIED_{type}_{date}"
    else:
        trainval_data_home = data_home+f"/trainval_data/UIED_{date}"
    test_data_home = data_home+"/test_data"
    # test_dir = "test_1122"
    make_dirs(trainval_data_home)
    # make_dirs(test_data_home)

    f_train = open(trainval_data_home+"/train.txt","w",encoding="utf-8")
    f_val = open(trainval_data_home+"/val.txt","w",encoding="utf-8")
    # f_test = open(test_data_home+"/test.txt",'w',encoding="utf-8")

    dirs = [ dir for dir in os.listdir(origin_data_home) if os.path.isdir(os.path.join(origin_data_home,dir))]
    # 训练验证数据集
    make_dirs(trainval_data_home+"/train_val/imgs")
    make_dirs(trainval_data_home+'/train_val/xmls')
    for dir in tqdm(dirs):
        dir_path = os.path.join(origin_data_home,dir)
        imgs = [img for img in os.listdir(os.path.join(dir_path,"imgs")) if os.path.splitext(img)[-1] in ['.jpg','.png']]
        for img in tqdm(imgs):
            img_name = os.path.splitext(img)[0]
            img_path = os.path.join(dir_path,"imgs",img)
            xml_path = os.path.join(dir_path,"xmls",img_name+".xml")
            
            # 重新命名，加入文件夹名，防止同名覆盖
            img_name = dir+"_"+img_name
            dst_img_path = trainval_data_home+"/train_val/imgs/"+img_name+".jpg"
            dst_xml_path = trainval_data_home+'/train_val/xmls/'+img_name+".xml"
            if not os.path.exists(img_path) or not os.path.exists(xml_path):
                continue
            # 测试集
            # make_dirs(test_data_home+f"/{test_dir}/imgs")
            # make_dirs(test_data_home+f"/{test_dir}/xmls")
            # dst_test_img_path = os.path.join(test_data_home+f"/{test_dir}/imgs/"+img)
            # dst_test_xml_path = os.path.join(test_data_home+f'/{test_dir}/xmls/'+img_name+".xml")
            if random.random() > 0.0:
                shutil.copy(img_path,dst_img_path)
                shutil.copy(xml_path,dst_xml_path)
                if random.random() > 0.1:
                    f_train.write(f"{dst_img_path.replace(trainval_data_home+'/','')} {dst_xml_path.replace(trainval_data_home+'/','')}\r")
                else:
                    f_val.write(f"{dst_img_path.replace(trainval_data_home+'/','')} {dst_xml_path.replace(trainval_data_home+'/','')}\r")
            else:
                # shutil.copy(img_path,dst_test_img_path)
                # shutil.copy(xml_path,dst_test_xml_path)
                # f_test.write(f"{dst_test_img_path} {dst_test_xml_path}\r")
                pass

    f_train.close()
    f_val.close()

def gen_label_list(date,data_home,type):
    '''
    生成标签列表
    args:
        date: 数据日期
        data_home: 数据根目录
        type: 检测类型，block还是元素
    return:
        None
    '''
    if type != "":
        xml_dir = f"{data_home}/trainval_data/UIED_{type}_{date}/train_val"
    else:
        xml_dir = f"{data_home}/trainval_data/UIED_{date}/train_val"
    f_labels = open(os.path.join(xml_dir,'../',"label_list.txt"),"w")
    xmls = os.listdir(xml_dir+"/xmls")
    clses = []      # 记录类别
    cls_nums = {}   # 统计每个类别数量
    for xml in xmls:
        xml_path = os.path.join(xml_dir+"/xmls",xml)
        tree,root = read_xml(xml_path)
        objs = tree.findall('object')
        for i,obj in enumerate(objs):
            cls_name = obj.find('name').text
            if cls_name in cls_nums:
                cls_nums[cls_name] += 1
            else:
                cls_nums[cls_name] = 1
            clses.append(cls_name)
        
    print("classes: ")
    for cls in set(clses):
        print(cls,cls_nums[cls])    # 类别 该类别样本数量 
        f_labels.write(cls+"\n")

def write_xml(folder: str, img_name: str, path: str, img_width: int, img_height: int, tag_num: int, tag_names: str, box_list:list,save_path:str,url_id:int=0):
    '''
    VOC标注xml文件生成函数
    args:
        folder: 文件夹名
        img_name: 图片的名字，不含后缀
        path: 图片完整路径   
        img_width: 图像宽
        img_height: 图像高
        tag_num: 图片内的标注框的数量
        tag_name: 类别名称
        box_list: 标注坐标,其数据格式为[[xmin1, ymin1, xmax1, ymax1],[xmin2, ymin2, xmax2, ymax2]....]
    return:
        None
    '''
    # 创建dom树对象
    doc = xml.dom.minidom.Document()
 
    # 创建root结点annotation，并用dom对象添加根结点
    root_node = doc.createElement("annotation")
    doc.appendChild(root_node)
 
    # 创建结点并加入到根结点
    folder_node = doc.createElement("folder")
    folder_value = doc.createTextNode(folder)
    folder_node.appendChild(folder_value)
    root_node.appendChild(folder_node)
 
    filename_node = doc.createElement("filename")
    filename_value = doc.createTextNode(img_name)
    filename_node.appendChild(filename_value)
    root_node.appendChild(filename_node)
 
    path_node = doc.createElement("path")
    path_value = doc.createTextNode(path)
    path_node.appendChild(path_value)
    root_node.appendChild(path_node)
 
    source_node = doc.createElement("source")
    database_node = doc.createElement("database")
    database_node.appendChild(doc.createTextNode("Unknown"))
    source_node.appendChild(database_node)
    root_node.appendChild(source_node)
 
    size_node = doc.createElement("size")
    for item, value in zip(["width", "height", "depth"], [img_width, img_height, 3]):
        elem = doc.createElement(item)
        elem.appendChild(doc.createTextNode(str(value)))
        size_node.appendChild(elem)
    root_node.appendChild(size_node)
 
    seg_node = doc.createElement("segmented")
    seg_node.appendChild(doc.createTextNode(str(0)))
    root_node.appendChild(seg_node)
 
    for i in range(tag_num):
        obj_node = doc.createElement("object")
        name_node = doc.createElement("name")
        name_node.appendChild(doc.createTextNode(tag_names[i]))
        obj_node.appendChild(name_node)
 
        pose_node = doc.createElement("pose")
        pose_node.appendChild(doc.createTextNode("Unspecified"))
        obj_node.appendChild(pose_node)
 
        trun_node = doc.createElement("truncated")
        trun_node.appendChild(doc.createTextNode(str(0)))
        obj_node.appendChild(trun_node)
 
        trun_node = doc.createElement("difficult")
        trun_node.appendChild(doc.createTextNode(str(0)))
        obj_node.appendChild(trun_node)
 
        bndbox_node = doc.createElement("bndbox")
        for item, value in zip(["xmin", "ymin", "xmax", "ymax"], box_list[i]):
            elem = doc.createElement(item)
            elem.appendChild(doc.createTextNode(str(value)))
            bndbox_node.appendChild(elem)
        obj_node.appendChild(bndbox_node)
        root_node.appendChild(obj_node)
 
    with open(os.path.join(save_path,"{}".format(img_name)+ ".xml"), "w", encoding="utf-8") as f:
        # writexml()第一个参数是目标文件对象
        # 第二个参数是根节点的缩进格式
        # 第三个参数是其他子节点的缩进格式
        # 第四个参数制定了换行格式
        # 第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")

def save_img_xml(height,img_path,xml_path,img,cls_names,box_list,dir_id):
    '''
    将网页长截图，裁剪成固定高度的图片，并取出对应的标签
    args:
        height: 子图的高度
        img_path: 图片保存路径
        xml_path: 标签保存路径
        img: 原始长截图数据
        cls_names: 类别标签列表
        box_list: 标注框列表
        dir_id: 当前长截图所在文件夹的名字，用于保存图片时区分文件夹
    return:
        None
    '''
    img_num = 0
    ratio = 0.3
    step = int(height * ratio)
    h,w = img.shape[:2]
    # 裁剪的个数
    total_num = h // step
    min_h = int(height * 2/3)

    while img_num < total_num:
        # 裁剪图片
        y_start = img_num*step
        y_end = img_num*step+height
        save_img = img[y_start : y_end,:]
        if save_img.shape[0] < min_h:
            break
        height = save_img.shape[0]          # 裁剪后可能会达不到1080的高度
        choosed_coords = []
        choosed_names = []
        edge_h_thresh = 5
        whr = 20
        min_h_whr = 25  # 与宽高比一起用
        minh = 10
        for name,coord in zip(cls_names,box_list):
            x1,y1,x2,y2 = coord
            # 坐标重新映射到[0,height]之间
            y1 -= y_start
            y2 -= y_start
            # 只保留当前页范围的元素

            if y2 < edge_h_thresh or y1 >= height-edge_h_thresh :
                continue
            # 边界裁剪
            if y1 < 0:
                y1 = 0
            if y2 >= height:
                y2 = height-1
            
            h_ = y2 - y1 + 1
            w_ = x2 - x1 + 1
            if h_ < min_h_whr and w_/h_ > whr or h_ < minh:
                continue
            
            choosed_coords.append([x1,y1,x2,y2])
            choosed_names.append(name)
        img_name = str(dir_id) + "_" + str(img_num)
        # 保存图片
        img_save_path = os.path.join(img_path,img_name+".jpg")
        cv2.imencode('.jpg', save_img)[1].tofile(img_save_path)
        # 保存xml
        write_xml(img_path,img_name,img_path,w,height,len(choosed_coords),choosed_names,choosed_coords,xml_path)
        img_num += 1
    print(f"{img_num} imgs saved in {img_path}, xmls saved in {xml_path}")

def cut_imgs_xmls(date,data_home,union_label="",exclude_labels = []):
    '''
    裁剪图片和标签并保存
    args:
        date: 数据日期
        data_hoeme: 数据根目录
        union_label: 是否使用同一个统一的标签，如全部全部都只当一个类别 
        exclude_labels: 哪些标签不用导出
    '''
    # data_home = "F:/Datasets/UIED"
    ori_labeld_path = os.path.join(data_home,f"原始标注数据/{date}")
    save_path = os.path.join(data_home,f"裁剪标注数据/{date}")
    dirs = os.listdir(ori_labeld_path)
    height = 1080
    for dir in dirs:
        # if dir in ["cunmin",'achong']:
        #     continue
        dir_path = os.path.join(ori_labeld_path,dir)
        img_folder = "imgs"
        try:
            imgs = os.listdir(dir_path+f"/{img_folder}")
        except:
            img_folder = "images"
            imgs = os.listdir(dir_path+f"/{img_folder}")
        save_dir_path = os.path.join(save_path,dir)
        img_save_path = os.path.join(save_dir_path,"imgs")
        xml_save_path = os.path.join(save_dir_path,"xmls")
        make_dirs(save_dir_path)
        make_dirs(img_save_path)
        make_dirs(xml_save_path)

        def func(img):
            pbar.update(1)
        # for img in tqdm(imgs[:100]):
            img_name = os.path.splitext(img)[0]
            img_path = os.path.join(dir_path+f"/{img_folder}",img)
            img_data = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            xml_path = os.path.join(dir_path+"/xmls",img_name+".xml")
            try:
                tree,root = read_xml(xml_path)
            except:
                print("error xml!")
                # continue
            objs = tree.findall('object')
            cls_list = []
            box_list = []
            for i,obj in enumerate(objs):
                cls_name = obj.find('name').text
                if union_label != "":
                    cls_name = "element"
                # 排除某些标签
                if cls_name in exclude_labels:
                    continue
                cls_list.append(cls_name)
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                box_list.append([xmin,ymin,xmax,ymax])
            save_img_xml(height,img_save_path,xml_save_path,img_data,cls_list,box_list,img_name)
        pbar = tqdm(total=len(imgs))
        pool = ThreadPool(processes=10)
        pool.map(func, imgs)
        pool.close()
        pool.join()
        pbar.close()

def pre_label_for_det(home,type):
    '''
    用训练好的模型做预标注
    args:
        home: 数据根目录
        type: 打标类型，属于区块还是元素
    return:
        None
    '''
    #=========== 初始化模型 =============
    if type == "":
        det_model_path = "weight/ppyoloe_plus_crn_m_80e_coco_UIED_0216.onnx"
        label_path = "weight/label_list.txt"
    else:
        det_model_path = "weight/ppyoloe_crn_s_p2_alpha_80e_UIED_ELE_0218.onnx"
        label_path = "weight/label_list_ELE.txt"

    det_net = PicoDet(model_pb_path = det_model_path,label_path = label_path,type=type)

    # 数据目录
    data_home = f"{home}/原始标注数据/2023_02_18/" 
    save_home = f"{home}/裁剪标注数据/2023_02_20/"
    dir = "cunmin2"
    imgs_path = os.path.join(data_home,dir,"imgs")        # 截的页面长图文件夹

    # 结果保存路径
    save_imgs_path = os.path.join(save_home,dir,"imgs")     # 裁剪出来的小图
    save_xmls_path = os.path.join(save_home,dir,"xmls")     # 裁剪出来的小图对应的标签
    make_dirs(save_imgs_path)
    make_dirs(save_xmls_path)

    height = 1080
    imgs = os.listdir(imgs_path)
    pbar = tqdm(total=len(imgs))

    def cut_save(img):
    # for img in tqdm(imgs):
        img_name = img.split(".")[0]
        img_path = os.path.join(imgs_path,img)
        try:
            # img = cv2.imdecode(img_path,cv2.IMREAD_COLOR)
            img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        except:
            # continue
            return
        pbar.update(1)
        # 裁剪
        img_num = 0
        ratio = 0.3
        # 每次向下走一步（这里步长设为一屏）
        step = int(height * ratio)
        h,w = img.shape[:2]
        # 裁剪的个数
        total_num = h // step
        min_h = int(height * 2/3)

        while img_num < total_num:
            # 裁剪图片
            y_start = img_num*step
            y_end = img_num*step+height
            save_img = img[y_start : y_end,:]
            im_h,im_w = save_img.shape[:2]
            # 裁剪出来的图片高度小于阈值不要
            if im_h < min_h:
                break
            det_results = det_net.infer(save_img)
            cls_list = []
            box_list = []
            for item in det_results:
                conf = item['confidence']
                if conf < 0.6:
                    continue
                cls_name = item["classname"]
                # cls_name = "ELE"
                box = item["box"]
                cls_list.append(cls_name)
                box_list.append(box)
                # print(conf,box)
            save_img_name = img_name+"_"+str(img_num)
            save_img_path = os.path.join(save_imgs_path,save_img_name+".jpg")
            # 保存图片和标签
            cv2.imencode('.jpg', save_img)[1].tofile(save_img_path)
            write_xml(save_img_path,save_img_name,save_img_path,im_w,im_h,len(cls_list),cls_list,box_list,save_xmls_path,0)
            img_num += 1

    # 创建多线程处理
    pbar = tqdm(total=len(imgs))
    pool = ThreadPool(processes=10)
    pool.map(cut_save, imgs)
    pool.close()
    pool.join()
    pbar.close()

def cut_imgs_for_cls(data_home,date):
    '''
    生成分类数据集
    args:
        data_home: 数据根目录
        date:   数据日期
    returns:
        None
    '''
    date_path = os.path.join(data_home,"原始数据",date)
    save_date_path = os.path.join(data_home,"裁剪数据",date)
    # save_date_path = os.path.join("F:/temp/",date)
    make_dirs(save_date_path)
    dirs = os.listdir(date_path)
    # 遍历每个文件夹
    for dir in tqdm(dirs):
        dir_path = os.path.join(date_path,dir)
        imgs_path = os.path.join(dir_path,"imgs")
        xmls_path = os.path.join(dir_path,"xmls")
        imgs = os.listdir(imgs_path)
        # 遍历每张图
        for img in tqdm(imgs):
            img_name = img.split('.')[0]
            img_path = os.path.join(imgs_path,img)
            img_data = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            h,w,c = img_data.shape
            xml_path = os.path.join(xmls_path,img_name+".xml")
            try:
                tree,root = read_xml(xml_path)
            except:
                print("error xml!")
                continue
            objs = tree.findall('object')
            box_list = []
            for i,obj in enumerate(objs):
                bndbox = obj.find('bndbox')
                delta = 3 # 外扩2个像素
                xmin = max(int(float(bndbox.find('xmin').text)) - delta,0)
                ymin = max(int(float(bndbox.find('ymin').text)) - delta,0)
                xmax = min(int(float(bndbox.find('xmax').text)) + delta,w-1)
                ymax = min(int(float(bndbox.find('ymax').text)) + delta,h-1)
                ROI = img_data[ymin:ymax,xmin:xmax]
                save_img_name = f"{dir}_{img_name}_{i}.jpg"
                img_save_path = os.path.join(save_date_path,save_img_name)
                cv2.imencode('.jpg', ROI)[1].tofile(img_save_path)

def gen_train_val_cls(data_home,date):
    '''
    生成分类训练和验证数据集
    args:
        data_home: 数据根目录
        date:   数据日期
    returns:
        None
    '''
    data_path = os.path.join(data_home,"分类数据",date).replace("\\","/")
    dirs = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,dir))]
    f_train = open(os.path.join(data_path,'train.txt'),"w",encoding="utf-8")
    f_val = open(os.path.join(data_path,"val.txt"),"w",encoding="utf-8")
    f_label = open(os.path.join(data_path,"labels.txt"),"w",encoding="utf-8")
    split = 0.8
    total_num = 1000

    # 生成训练和验证txt文件
    for i,dir in tqdm(enumerate(dirs)):
        cls_name = dir.split('_')[-1]
        f_label.write(f"{i} {cls_name}\n")
        dir_path = os.path.join(data_path,dir)
        imgs = [img for img in os.listdir(dir_path) if os.path.splitext()[-1] in ['.jpg','.png']]
        img_num = len(imgs)
        random.shuffle(imgs)
        # 图片数不足的，重复取
        if img_num < total_num:
            times = total_num // img_num
            rest = total_num % img_num
            imgs = imgs*times + imgs[:rest]
        for img in imgs[:total_num]:
            img_path = os.path.join(dir_path,img).replace('\\','/').replace(data_path+"/","")
            if random.random() < split:
                f_train.write(img_path+" "+str(i)+"\n")
            else:
                f_val.write(img_path+" "+str(i)+"\n")
    f_train.close()
    f_val.close()
    f_label.close()
    print("done!")

def pre_label_cls(data_home):
    '''
    用预训练分类模型做预标注
    args:
        data_home: 数据根目录
    '''
    cls_model_path = "weight/ResNet50_5_1000.onnx"
    label_list_path = "weight/label_list_cls.txt"
    # 初始化分类模型
    cls_net = ResNet(model_pb_path = cls_model_path,label_path = label_list_path)
    save_path = data_home+"_out"
    make_dirs(save_path)
    imgs = os.listdir(data_home)
    for img in tqdm(imgs):
        img_path = os.path.join(data_home,img)
        try:
            img_data = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            result = cls_net.cls_onnx(img_data)
            cls_path = os.path.join(save_path,result["label"])
            make_dirs(cls_path)
            shutil.copy(img_path,os.path.join(cls_path,img))
            print(result)

        except Exception as e:
            print(e)
            continue
    print(f"img saved in {save_path}")

def pre_label_for_multicls_det():
    '''
    用预训练的分类模型对目标检测的类别标签做预标注
    '''
    # 初始化分类模型
    cls_model_path = "weight/ResNet50_5_mc_1000.onnx"
    label_list_path = "weight/label_list_cls.txt"
    cls_net = ResNet(model_pb_path = cls_model_path,label_path = label_list_path)
    data_home  = "Y:/zx-AI_lab/RPA/页面元素检测/元素分类/原始数据/2023_02_27"

    save_home = "Y:/zx-AI_lab/RPA/页面元素检测/元素检测/原始数据/2023_02_27"

    dirs = os.listdir(data_home)
    for dir in dirs:
        print(dir)
        dir_path = os.path.join(data_home,dir)
        imgs_path = os.path.join(dir_path,"imgs")
        xmls_path = os.path.join(dir_path,"xmls")
        save_dir = os.path.join(save_home,dir)
        make_dirs(save_dir)
        imgs = [img for img in os.listdir(imgs_path) if os.path.splitext(img)[-1] in ['.jpg','.png']]
        for img in tqdm(imgs):
            img_name = os.path.splitext(img)[0]
            img_path = os.path.join(imgs_path,img)
            try:
                img_data = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
                h,w,c = img_data.shape
            except Exception as e:
                print(e)
                continue
            
            # 读取xml标注文件
            xml_path = os.path.join(xmls_path,img_name+".xml")
            try:
                tree,root = read_xml(xml_path)
            except Exception as e:
                print(e)
                continue
            objs = tree.findall('object')
            box_list = []
            
            # 预测并修改box的类比
            def modify_label(obj):
            # for i,obj in enumerate(objs):
                pbar.update(1)
                bndbox = obj.find('bndbox')
                delta = 3 # 外扩2个像素
                xmin = max(int(float(bndbox.find('xmin').text)) - delta,0)
                ymin = max(int(float(bndbox.find('ymin').text)) - delta,0)
                xmax = min(int(float(bndbox.find('xmax').text)) + delta,w-1)
                ymax = min(int(float(bndbox.find('ymax').text)) + delta,h-1)
                ROI = img_data[ymin:ymax,xmin:xmax]
                # 分类
                result = cls_net.cls_onnx(ROI)
                obj.find('name').text = result["label"]
            
            pbar = tqdm(total=len(objs))
            pool = ThreadPool(processes=10)
            pool.map(modify_label, objs)
            pool.close()
            pool.join()
            pbar.close()

            # 保存为新的xml
            save_imgs_path = os.path.join(save_dir,"imgs")
            save_xmls_path = os.path.join(save_dir,"xmls")
            make_dirs(save_imgs_path)
            make_dirs(save_xmls_path)
            save_img_path = os.path.join(save_imgs_path,img)
            shutil.copy(img_path,save_img_path)
            save_xml_path = save_xmls_path+"/"+img_name+".xml"
            tree.write(save_xml_path, encoding="utf-8",xml_declaration=True)

def pre_label_for_block_ELEs(data_home,save_path):
    '''
    用训练好的模型做block和elements检测，用于整合输出
    args:
        data_home: 数据根目录
        save_path: 打标类型，属于区块还是元素
    returns:
        None
    '''
    #=========== 初始化模型 =============
    
    block_det_model_path = "weight/ppyoloe_plus_crn_m_80e_coco_UIED_0220.onnx"
    block_label_path = "weight/label_list.txt"
   
    ELE_det_model_path = "weight/ppyoloe_crn_s_p2_alpha_80e_UIED_ELE_0227.onnx"
    ELE_label_path = "weight/label_list_ELE.txt"

    block_det_net = PicoDet(model_pb_path = block_det_model_path,label_path = block_label_path)
    ELE_det_net = PicoDet(model_pb_path = ELE_det_model_path,label_path = ELE_label_path,type="ELE")

    # 数据目录
    make_dirs(save_path)
    dirs = [dir for dir in os.listdir(data_home) if os.path.isdir(os.path.join(data_home,dir))]
    
    for dir in dirs:
        imgs_path = os.path.join(data_home,dir,"imgs")        # 截的页面长图文件夹
        # 结果保存路径
        save_imgs_path = os.path.join(save_path,dir,"imgs")     # 裁剪出来的小图
        save_xmls_path = os.path.join(save_path,dir,"xmls")     # 裁剪出来的小图对应的标签
        make_dirs(save_imgs_path)
        make_dirs(save_xmls_path)

        height = 1080
        imgs = os.listdir(imgs_path)
        pbar = tqdm(total=len(imgs))

        def cut_save(img):
        # for img in tqdm(imgs):
            img_name = img.split(".")[0]
            img_path = os.path.join(imgs_path,img)
            try:
                # img = cv2.imdecode(img_path,cv2.IMREAD_COLOR)
                img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            except:
                # continue
                return
            pbar.update(1)
            # 裁剪
            img_num = 0
            ratio = 0.3
            # 每次向下走一步（这里步长设为一屏）
            step = int(height * ratio)
            h,w = img.shape[:2]
            # 裁剪的个数
            total_num = h // step
            min_h = int(height * 2/3)

            while img_num < total_num:
                # 裁剪图片
                y_start = img_num*step
                y_end = img_num*step+height
                save_img = img[y_start : y_end,:]
                im_h,im_w = save_img.shape[:2]
                # 裁剪出来的图片高度小于阈值不要
                if im_h < min_h:
                    break
                # block检测
                block_det_results = block_det_net.infer(save_img)
                cls_list = []
                box_list = []
                for item in block_det_results:
                    conf = item['confidence']
                    if conf < 0.6:
                        continue
                    cls_name = item["classname"]
                    # cls_name = "ELE"
                    box = item["box"]
                    cls_list.append(cls_name)
                    box_list.append(box)
                # 元素检测
                ELE_det_results = ELE_det_net.infer(save_img)
                for item in ELE_det_results:
                    conf = item['confidence']
                    if conf < 0.6:
                        continue
                    cls_name = item["classname"]
                    # cls_name = "ELE"
                    box = item["box"]
                    cls_list.append(cls_name)
                    box_list.append(box)
                
                save_img_name = img_name+"_"+str(img_num)
                save_img_path = os.path.join(save_imgs_path,save_img_name+".jpg")
                # # 保存图片和标签
                # cv2.imencode('.jpg', save_img)[1].tofile(save_img_path)
                # write_xml(save_img_path,save_img_name,save_img_path,im_w,im_h,len(cls_list),cls_list,box_list,save_xmls_path,0)
                # img_num += 1

        # 创建多线程处理
        pbar = tqdm(total=len(imgs))
        pool = ThreadPool(processes=10)
        pool.map(cut_save, imgs)
        pool.close()
        pool.join()
        pbar.close()

def infer_long(type ,test_path):
    '''
    推理测试长图（高度大于屏幕高度的长截图）
    args:
        type: 推理的任务类型，区块还是元素
        test_path: 测试图片路径
    returns:
        None
    '''
    #=========== 初始化模型 =============
    if type == "":
        det_model_path = "weight/ppyoloe_plus_crn_m_80e_coco_UIED_0216.onnx"
        label_path = "weight/label_list.txt"
    else:
        det_model_path = "weight/ppyoloe_crn_s_p2_alpha_80e_UIED_ELE_0218.onnx"
        label_path = "weight/label_list_ELE.txt"

    det_net = PicoDet(model_pb_path = det_model_path,label_path = label_path,type = type)

    test_imgs = os.listdir(test_path)
    test_out = os.path.join(test_path,'../',test_path+"_out")
    make_dirs(test_out)
    for img in test_imgs:
        img_name = img.split('.')[0]
        img_path = os.path.join(test_path,img)
        try:
            # img = cv2.imdecode(img_path,cv2.IMREAD_COLOR)
            img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        except:
            continue
            # return
        img_num = 0
        ratio = 0.3
        height = 1080
        # 每次向下走一步（这里步长设为一屏）
        step = int(height * ratio)
        h,w = img.shape[:2]
        # 裁剪的个数
        total_num = h // step
        min_h = int(height * 2/3)
        while img_num < total_num:
            # 裁剪图片
            y_start = img_num*step
            y_end = img_num*step+height
            save_img = img[y_start : y_end,:].copy()
            im_h,im_w = save_img.shape[:2]
            # 裁剪出来的图片高度小于阈值不要
            if im_h < min_h:
                break
            det_results = det_net.infer(save_img)
            cls_list = []
            box_list = []
            for item in det_results:
                conf = item['confidence']
                if conf < 0.6:
                    continue
                cls_name = item["classname"]
                # cls_name = "ELE"
                box = item["box"]
                cls_list.append(cls_name)
                box_list.append(box)
                cv2.rectangle(save_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)

            save_img_path = os.path.join(test_out,img_name+"_"+str(img_num)+".jpg")
            cv2.imencode('.jpg', save_img)[1].tofile(save_img_path)
            img_num += 1 

def infer_short(type,test_path):
    '''
    推理测试短图（高度小于等于屏幕高度）
    args:
        type: 推理的任务类型，区块还是元素
        test_path: 测试图片路径
    returns:
        None
    '''
    #=========== 初始化模型 =============
    if type == "":
        det_model_path = "weight/ppyoloe_plus_crn_m_80e_coco_UIED_0216.onnx"
        label_path = "weight/label_list.txt"
    else:
        det_model_path = "weight/ppyoloe_crn_s_p2_alpha_80e_UIED_ELE.onnx"
        label_path = "weight/label_list_ELE.txt"
    det_net = PicoDet(model_pb_path = det_model_path,label_path = label_path,type=type)

    test_imgs = os.listdir(test_path)
    test_out = os.path.join(test_path,'../',test_path+"_out")
    make_dirs(test_out)
    for img in test_imgs:
        img_name = img.split('.')[0]
        img_path = os.path.join(test_path,img)
        try:
            # img = cv2.imdecode(img_path,cv2.IMREAD_COLOR)
            img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        except:
            continue
            # return
        
        det_results = det_net.infer(img)
        cls_list = []
        box_list = []
        draw_img = img.copy()
        for item in det_results:
            conf = item['confidence']
            if conf < 0.6:
                continue
            cls_name = item["classname"]
            # cls_name = "ELE"
            box = item["box"]
            cls_list.append(cls_name)
            box_list.append(box)
            cv2.rectangle(draw_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)

        save_img_path = os.path.join(test_out,img_name+"_"+str(img_num)+".jpg")
        cv2.imencode('.jpg', draw_img)[1].tofile(save_img_path)
        img_num += 1 

def extract_merge_info(info):
    '''
    合并block和元素信息
    args:
        info: 所有检测结果列表
    return:
        merge_info: 合并后的层级关系信息
    '''
    th = 0.9
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
                if overlap>=th and iou<1.0:
                    # print(iou,overlap)
                    sum_tree+=1
                    sum_index.append(j)
        
        cal_info.append((name,xmin,ymin,xmax,ymax,sum_tree,sum_index))



    merge_info = []

    ## 获取树的最大层数
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

def process_merge_info(merge_info):
    '''
    根据合并后的层级关系信息，做后处理，对信息进行结构化
    args:
        merge_info: 合并后的层级关系信息
    return:
        result: 组织好的结构化信息
    '''
    # 计算树的最大层数

    tree_depth = [ele[5] for ele in merge_info]
    max_tree_depth = np.max(tree_depth)

    new_info = []
    for item in merge_info:
        contains=[]
        (name,xmin,ymin,xmax,ymax,sum_tree,tree_index) = item
        new_info.append((name,xmin,ymin,xmax,ymax,sum_tree,contains))
    
    for j in range(max_tree_depth-1,-1,-1):
        for i in range(len(merge_info)):
            item = merge_info[i]
            (name,xmin,ymin,xmax,ymax,sum_tree,tree_index) = item
            if sum_tree==j:
                if(len(tree_index)>0):
                    for index in tree_index:
                        new_info[i][6].append((new_info[index]))
    
    result = []
    for i in range(len(new_info)):
        item = new_info[i]
        name,xmin,ymin,xmax,ymax,sum_tree,contains = item
        if sum_tree==0:
            result.append(item)
    return result

def merge_info(info):
    '''
    合并block和元素信息，并做结构化
    args:
        info: 所有box列表
    returns:
        merge_info: 结构化信息
    '''
    merge_info = extract_merge_info(info)
    merge_info = process_merge_info(merge_info)
    return merge_info

def infer_block_ELEs(data_home,save_path,gen_json = False):
    '''
    用训练好的模型做block和elements检测推理
    args:
        test_home: 数据根目录
        save_path: 打标类型，属于区块还是元素
        gen_json: 是否生成json文件,json文件与图片同路径
    returns:
        None
    '''
    #=========== 初始化模型 =============
    
    block_det_model_path = "weight/ppyoloe_plus_crn_m_80e_coco_UIED_0220.onnx"
    block_label_path = "weight/label_list.txt"
   
    ELE_det_model_path = "weight/ppyoloe_crn_s_p2_alpha_80e_UIED_ELE_0228_paste.onnx"
    ELE_label_path = "weight/label_list_ELE_0228.txt"

    block_det_net = PicoDet(model_pb_path = block_det_model_path,label_path = block_label_path)
    ELE_det_net = PicoDet(model_pb_path = ELE_det_model_path,label_path = ELE_label_path,type="ELE")

    # 数据目录
    make_dirs(save_path)
    # 短图文件夹
    imgs = [img for img in os.listdir(data_home) if os.path.isfile(os.path.join(data_home,img))]        

    def func(img):
    # for img in imgs:
        img_name = img.split(".")[0]
        img_path = os.path.join(data_home,img)
        try:
            # img = cv2.imdecode(img_path,cv2.IMREAD_COLOR)
            img_data= cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        except:
            # continue
            return
        pbar.update(1)
        
        im = Image.fromarray(img_data[:,:,::-1])

        info = []

        # block检测
        block_det_results = block_det_net.infer(img_data)
        box_list = []
        for item in block_det_results:
            conf = item['confidence']
            if conf < 0.6:
                continue
            cls_name = item["classname"]
            cls_id = item['classid']
            # cls_name = "ELE"
            box = item["box"]
            box_list.append([cls_id,conf,*box])
            info.append([cls_name,*box])

        # 合并5个类别
        labels = block_det_net.classes+ELE_det_net.classes
        if len(box_list) > 0:
            im = draw_box(im,np.array(box_list),labels=labels)
        
        box_list = []
        # 元素检测
        ELE_det_results = ELE_det_net.infer(img_data)
        for item in ELE_det_results:
            conf = item['confidence']
            if conf < 0.6:
                continue
            cls_name = item["classname"]
            cls_id = item['classid'] + 1
            # cls_name = "ELE"
            box = item["box"]
            box_list.append([cls_id,conf,*box])
            info.append([cls_name,*box])

        if len(box_list) > 0:
            im = draw_box(im,np.array(box_list),labels=labels)
            save_img_path = os.path.join(save_path,img)
            im.save(save_img_path)
            # 是否生成json,保存路径与图片同目录
            if gen_json:
                merged_info = merge_info(info)
                with open(os.path.join(save_path,f"{img_name}.json"),'w',encoding='utf-8') as f:
                    json.dump(merged_info,f)

    # 创建多线程处理
    pbar = tqdm(total=len(imgs))
    pool = ThreadPool(processes=10)
    pool.map(func, imgs)
    pool.close()
    pool.join()
    pbar.close()

def match():
    '''
    用模板匹配的方法辅助对数据做分类
    '''
    tmplates_path = "F:/temp/templates_ico"
    tmps = os.listdir(tmplates_path)
    tmps_data = []
    for tmp in tmps:
        tmp_path = os.path.join(tmplates_path,tmp)
        tmp_data = cv2.imread(tmp_path,cv2.IMREAD_COLOR)
        tmps_data.append(tmp_data)

    method = "cv2.TM_CCOEFF"
    method = 'cv2.TM_SQDIFF_NORMED'
    method = "cv2.TM_CCOEFF_NORMED"         # 值越大越好 
    test_path = "F:/temp/2023_02_21"
    save_path = "F:/temp/2023_02_21_ico"
    # save_path = "F:/temp/2023_02_21_按钮1"
    make_dirs(save_path)
    test_imgs = os.listdir(test_path)

    def func(img):
        pbar.update(1)
    # for img in tqdm(test_imgs):
        img_path = os.path.join(test_path,img)
        img_data = cv2.imread(img_path,cv2.IMREAD_COLOR)
        imh,imw,c = img_data.shape
        min_sim = 10000
        max_sim = -10000
        for i,tmp in enumerate(tmps_data):
            tmph,tmpw,tmpc = tmp.shape
            if imh < tmph -5 or imw < tmpw -5 :
                continue
            img_data = cv2.resize(img_data,(tmpw,tmph))
            res = cv2.matchTemplate(img_data, tmp, eval(method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < min_sim:
                min_sim = min_val
            if max_val > max_sim:
                max_sim = max_val

        # if min_sim < 0.008:
        if max_sim > 0.2:
            shutil.move(img_path,os.path.join(save_path,img))
    
    pbar = tqdm(total=len(test_imgs))
    pool = ThreadPool(processes=10)
    pool.map(func, test_imgs)
    pool.close()
    pool.join()
    pbar.close()

def random_paste_items(train_data_path, labels_path ,labels=[],paste_num = 5):
    '''
    针对部分类别占比较少的元素，使用随机重复粘贴的方式来增加样本数
    args:
        train_data_path: 训练数据路径
        labels_path: 分好类的标签文件夹
        labels: 要粘贴的标签类别
        paste_num: 随机贴几个
    returns:
        None
    '''
    save_path = os.path.dirname(train_data_path)+"_paste"
    img_save_dir = os.path.join(save_path,"trainval/imgs")
    xml_save_dir = os.path.join(save_path,"trainval/xmls")
    make_dirs(save_path)
    make_dirs(img_save_dir)
    make_dirs(xml_save_dir)
    imgs_path = os.path.join(train_data_path,"imgs")
    xmls_path = os.path.join(train_data_path,"xmls")
    imgs = [img for img in os.listdir(imgs_path) if os.path.splitext(img)[-1] in ['.jpg','.png']]
    date = "2023_02_21"
    f_train = open(os.path.join(save_path,"train.txt"),"w")
    f_val = open(os.path.join(save_path,"val.txt"),"w")
    def func(img):
        pbar.update(1)
        img_path = os.path.join(imgs_path,img)
        img_name = os.path.splitext(img)[0]
        try:
            # img = cv2.imdecode(img_path,cv2.IMREAD_COLOR)
            img_data= cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
        except:
            # continue
            return
        imgh,imgw,_ = img_data.shape
        xml_path = os.path.join(xmls_path,img_name+".xml")
        tree,root = read_xml(xml_path)
        objs = tree.findall('object')
        cls_list = []
        box_list = []
        for i,obj in enumerate(objs):
            cls_name = obj.find('name').text
            cls_list.append(cls_name)
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            box_list.append([xmin,ymin,xmax,ymax])
        box_list = np.array(box_list)
        # 遍历所有要贴的标签
        for label in labels:
            label_path = os.path.join(labels_path,date+"_"+label)
            labelimgs = [img for img in os.listdir(label_path) if os.path.splitext(img)[-1] in ['.jpg','.png']]
            random.shuffle(labelimgs)
            num = 0
            while num < paste_num:
                label_img = random.choice(labelimgs)
                label_img_data = cv2.imdecode(np.fromfile(os.path.join(label_path,label_img),dtype=np.uint8),-1)
                h,w,c = label_img_data.shape
                delta = 3
                lxmin = random.randint(0,imgw - w)
                lymin = random.randint(0,imgh - h)
                lxmax = lxmin + w
                lymax = lymin + h
                
                # 判断是否空白位置
                ltcor = ((box_list[:,0] <= lxmin) & (lxmin <= box_list[:,2])) & ((box_list[:,1] <= lymin) & (lymin< box_list[:,3]))
                rbcor = ((box_list[:,0] <= lxmax) & (lxmin <= box_list[:,2])) & ((box_list[:,1] <= lymax) & (lymin< box_list[:,3]))
                # 背景纯色
                bg = img_data[lymin:lymax,lxmin:lxmax]
                pure = (bg.sum() == bg[0,0].sum() * bg.shape[0]*bg.shape[1])
                if ltcor.sum()== 0 and rbcor.sum() == 0 and pure :
                    
                    img_data[lymin:lymax,lxmin:lxmax] = label_img_data
                    node_object = subElement(root, 'object')            #在根节点node_root下面添加名为'object'的子节点
                    node_object_name = subElement(node_object, 'name')  #在根节点node_root的子节点object下面继续添加子节点object的子节点'name'
                    node_object_name.text = label                       #设定该节点的文本text信息
                    node_object_pose = subElement(node_object, 'pose')
                    node_object_pose.text = "Unspecified"
                    node_object_truncated = subElement(node_object, 'truncated')
                    node_object_truncated.text = '%s' % int(0)
                    node_object_difficult = subElement(node_object, 'difficult')
                    node_object_difficult.text = '%s' % int(0)
                    # object坐标
                    node_bndbox = subElement(node_object, 'bndbox')
                    node_xmin = subElement(node_bndbox, 'xmin')
                    node_xmin.text = '%s' % int(lxmin + delta)
                    node_ymin = subElement(node_bndbox, 'ymin')
                    node_ymin.text = '%s' % int(lymin + delta)
                    node_xmax = subElement(node_bndbox, 'xmax')
                    node_xmax.text = '%s' % int(lxmax - delta)
                    node_ymax = subElement(node_bndbox, 'ymax')
                    node_ymax.text = '%s' % int(lymax - delta)
                    labelbox = np.array([int(lxmin+delta),int(lymin+delta),int(lxmax+delta),int(lymax+delta)]).reshape(1,-1)
                    box_list = np.concatenate((box_list,labelbox),0)
                    num +=1

        img_save_path = os.path.join(img_save_dir,img).replace("\\","/")
        xml_save_path =os.path.join(xml_save_dir,img_name+".xml").replace("\\",'/')
        cv2.imencode('.jpg', img_data)[1].tofile(img_save_path)
        xml = tostring(root, pretty_print=True) #将修改后的xml节点信息导出
        with open(xml_save_path,'wb') as f: #将修改后的xml节点信息覆盖掉修改前的
            f.write(xml)
        if random.random() < 0.8:
            f_train.write(f"{img_save_path.replace(save_path+'/','')} {xml_save_path.replace(save_path+'/','')}\r")
        else:
            f_val.write(f"{img_save_path.replace(save_path+'/','')} {xml_save_path.replace(save_path+'/','')}\r")
    
    # 复制标签列表
    shutil.copy(os.path.join(os.path.dirname(train_data_path),"label_list.txt"),os.path.join(save_path,"label_list.txt"))
    # 创建多线程处理
    pbar = tqdm(total=len(imgs))
    pool = ThreadPool(processes=10)
    pool.map(func, imgs)
    pool.close()
    pool.join()
    pbar.close()

if __name__ == "__main__":
    
    #=============== UIED原始数据处理和训练数据制作 ===================
    home = "F:/Datasets/UIED/block检测"
    type = ""
    # home = "F:/Datasets/UIED/元素检测"
    # type = "ELE"
    date= "2023_02_20"
    # 长图裁剪
    # cut_imgs_xmls(date,home)
    # # 生成训练验证集
    # gen_train_val2(date,home,type)
    # # 生成标签列表
    # gen_label_list(date,home,type)

    #============= 用训练好的检测模型做预标注 ==========================
    # pre_label_for_det(home,type)

    #============= 训练好的模型直接推理看效果 ==========================
    type = "ELE"
    test_path = "F:/Datasets/UIED/测试/中文页面检测"
    # infer_long(type,test_path)
    # infer_short(test_path)

    #============= 裁剪图片，用于分类 ==========================
    data_home = "F:/Datasets/UIED/元素分类"
    date = "2023_02_21"
    # cut_imgs_for_cls(data_home,date)
    # gen_train_val_cls(data_home,date)

    #============= 用训练好的分类模型做分类预测 ==========================
    data_home = "F:/Datasets/UIED/元素分类/裁剪数据/2023_02_21"
    # pre_label_cls(data_home)

    #============= 用训练好的分类模型给检测数据打标 ==========================
    # 用模板匹配的方法自动粗分类
    # match()

    # 单类别长图打标成多类别
    # pre_label_for_multicls_det()

    # 长图裁剪成短图用于训练
    home = "F:/Datasets/UIED/元素检测"
    # 指定哪些标签要排除
    exclude_labels = ["text"]
    date = "2023_02_21"
    # 长图裁剪成短图
    # cut_imgs_xmls(date,home,exclude_labels=exclude_labels)
    
    # #生成检测训练数据
    # date = "2023_02_21"
    # type= "ELE"
    # gen_train_val2(date,home,type)
    # # # 生成标签列表
    # gen_label_list(date,home,type)

    #============= 用训练好的block检测模型和多类元素检测模型做预测 =================
    data_home = "F:/Datasets/UIED/元素检测/原始标注数据/2023_02_21"
    save_path = data_home+"_block_ELE"
    # pre_label_for_block_ELEs(data_home,save_path)

    # 对裁剪好的图做预测
    data_home = "F:/Datasets/UIED/元素检测/裁剪标注数据/2023_02_21/achong/imgs"
    save_path = "F:/Datasets/UIED/元素检测/测试结果/achong_block_ELE_infer_noconf_0228_paste"

    # data_home = "F:/Datasets/UIED/tmp"
    # save_path = "F:/Datasets/UIED/tmp/block_ELE_infer_noconf"
    infer_block_ELEs(data_home,save_path,True) 

    #=============== 粘贴数据增强 =============================

    trainval = "F:/Datasets/UIED/元素检测/trainval_data/UIED_ELE_2023_02_21/train_val"
    labels_path = "F:/Datasets/UIED/元素分类/分类数据/2023_02_21"
    labels = ['select']
    # random_paste_items(trainval,labels_path,labels)