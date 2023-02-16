
import xml.etree.ElementTree as ET
import os
import random
import cv2
from test_get_img import write_xml
from tqdm import tqdm
import numpy as np
import shutil
from picodet import PicoDet

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
    
def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gen_train_val():
    '''
    生成训练验证数据集，没有子文件夹
    '''
    data_home = "F:/Datasets/UIED/tmp2"
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


def gen_train_val2(date):
    '''
    生成训练验证数据集,多个子文件夹
    '''
    data_home = "F:/Datasets/UIED"
    origin_data_home = data_home+f"/裁剪标注数据/{date}"
    trainval_data_home = data_home+f"/trainval_data/UIED_{date}"
    test_data_home = data_home+"/test_data"
    # test_dir = "test_1122"
    make_dirs(trainval_data_home)
    # make_dirs(test_data_home)

    f_train = open(trainval_data_home+"/train.txt","w",encoding="utf-8")
    f_val = open(trainval_data_home+"/val.txt","w",encoding="utf-8")
    # f_test = open(test_data_home+"/test.txt",'w',encoding="utf-8")

    dirs = os.listdir(origin_data_home)
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
                    f_train.write(f"{dst_img_path.replace(trainval_data_home,'')} {dst_xml_path.replace(trainval_data_home,'')}\r")
                else:
                    f_val.write(f"{dst_img_path.replace(trainval_data_home,'')} {dst_xml_path.replace(trainval_data_home,'')}\r")
            else:
                # shutil.copy(img_path,dst_test_img_path)
                # shutil.copy(xml_path,dst_test_xml_path)
                # f_test.write(f"{dst_test_img_path} {dst_test_xml_path}\r")
                pass

    f_train.close()
    f_val.close()

def gen_label_list(date):
    '''
    生成标签列表
    '''
    xml_dir = f"F:/Datasets/UIED/trainval_data/UIED_{date}/train_val"
    f_labels = open(os.path.join(xml_dir,'../',"label_list.txt"),"w")
    xmls = os.listdir(xml_dir+"/xmls")
    clses = [] 
    for xml in xmls:
        xml_path = os.path.join(xml_dir+"/xmls",xml)
        tree,root = read_xml(xml_path)
        objs = tree.findall('object')
        for i,obj in enumerate(objs):
            cls_name = obj.find('name').text
            clses.append(cls_name)
        
    print("classes: ")
    for cls in set(clses):
        print(cls)
        f_labels.write(cls+"\n")

def save_img_xml(height,img_path,xml_path,img,cls_names,box_list,dir_id):
    '''
    裁剪一块块的图片，并取出对应的标签
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

def cut_imgs_xmls(date):
    '''
    裁剪图片和标签并保存
    '''
    data_home = "F:/Datasets/UIED"
    ori_labeld_path = os.path.join(data_home,f"原始标注数据/{date}")
    save_path = os.path.join(data_home,f"裁剪标注数据/{date}")
    dirs = os.listdir(ori_labeld_path)
    height = 1080
    for dir in dirs:
        if dir in ["cunmin"]:
            continue
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
        for img in tqdm(imgs):
            img_name = os.path.splitext(img)[0]
            img_path = os.path.join(dir_path+f"/{img_folder}",img)
            img_data = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
            xml_path = os.path.join(dir_path+"/xmls",img_name+".xml")
            try:
                tree,root = read_xml(xml_path)
            except:
                print("error xml!")
                continue
                
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
            save_img_xml(height,img_save_path,xml_save_path,img_data,cls_list,box_list,img_name)

def pre_label():
    '''
    用训练好的模型做预标注
    '''
    #=========== 初始化模型 =============
    det_model_path = "weight/ppyoloe_plus_crn_m_80e_coco_UIED_opt.onnx"
    label_path = "weight/label_list.txt"
    det_net = PicoDet(model_pb_path = det_model_path,label_path = label_path)

    # 数据目录
    data_home = "F:/Datasets/UIED/trainval_data/UIED_2023_02_15/test2/" 
    save_home = "F:/Datasets/UIED/trainval_data/UIED_2023_02_15/test2/"
    imgs_path = os.path.join(data_home,"screen")        # 截的页面长图文件夹

    # 结果保存路径
    save_imgs_path = os.path.join(save_home,"imgs")     # 裁剪出来的小图
    save_xmls_path = os.path.join(save_home,"xmls")     # 裁剪出来的小图对应的标签
    make_dirs(save_imgs_path)
    make_dirs(save_xmls_path)

    height = 1080
    imgs = os.listdir(imgs_path)
    for img in tqdm(imgs):
        img_name = img.split(".")[0]
        img_path = os.path.join(imgs_path,img)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        
        # 裁剪
        img_num = 0
        ratio = 1
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
                cls_name = item["classname"]
                box = item["box"]
                cls_list.append(cls_name)
                box_list.append(box)
            save_img_name = img_name+"_"+str(img_num)
            save_img_path = os.path.join(save_imgs_path,save_img_name+".jpg")
            # 保存图片和标签
            cv2.imwrite(save_img_path,save_img)
            write_xml(save_img_path,save_img_name,save_img_path,im_w,im_h,len(cls_list),cls_list,box_list,save_xmls_path)
            img_num += 1
            

if __name__ == "__main__":

    
    #=============== UIED原始数据处理和训练数据制作 ===================
    
    # date= "2023_02_15"
    # # 长图裁剪
    # cut_imgs_xmls(date)
    # # 生成训练验证集
    # gen_train_val2(date)
    # # 生成标签列表
    # gen_label_list(date)

    #============= 用训练好的检测模型做预标注 ==========================
    pre_label()

