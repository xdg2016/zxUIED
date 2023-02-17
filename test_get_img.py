from bs4 import BeautifulSoup
from pprint import pprint
import pyautogui
#引入selenium库中的 webdriver 模块
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.remote_connection import LOGGER
#引入time库
import time
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
import random
import numba as nb
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

def get_tag_elements(driver,tag_names):
    '''
    根据html标签名查找
    '''
    all_results = []
    for tag in tqdm(tag_names):
        # results = driver.find_elements(By.TAG_NAME,tag)
        results = [ele for ele in driver.find_elements(By.TAG_NAME,tag) if ele.is_displayed()]
        # hidden_results = driver.find_elements(By.XPATH, f"//*[contains(@style,'hidden')]")
        # results = list(set(results) - set(hidden_results))
        all_results.extend(results)
    return all_results

def get_class_contains_elements(driver,classes):
    '''
    根据类名模糊匹配的方式查找
    '''
    all_results = []
    for cls in tqdm(classes):
        results =  driver.find_elements(By.XPATH, f"//*[contains(@class,'{cls}')]")
        all_results += results
    return all_results

def draw(img,results,save_path = "anna.png"):
    '''
    将解析的结果画在图上
    '''
    for e in results:
        try:
            # print(e.rect)
            x,y,w,h = int(e.rect['x']),int(e.rect['y']),int(e.rect['width']),int(e.rect['height'])
            pt1 = x,y
            pt2 = x+w,y+h
            cv2.rectangle(img,pt1,pt2,color=(0,0,255),thickness=2)
            cv2.putText(img, e.tag_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        except:
            continue

    # cv2.imshow("img",img_)
    cv2.imwrite(save_path,img)
    cv2.waitKey(0)

def write_xml(folder: str, img_name: str, path: str, img_width: int, img_height: int, tag_num: int, tag_names: str, box_list:list,save_path:str):
    '''
    VOC标注xml文件生成函数
    :param folder: 文件夹名
    :param img_name:
    :param path:
    :param img_width:
    :param img_height:
    :param tag_num: 图片内的标注框数量
    :param tag_name: 标注名称
    :param box_list: 标注坐标,其数据格式为[[xmin1, ymin1, xmax1, ymax1],[xmin2, ymin2, xmax2, ymax2]....]
    :return: a standard VOC format .xml file, named "img_name.xml"
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
 
    with open(os.path.join(save_path,img_name+ ".xml"), "w", encoding="utf-8") as f:
        # writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")

def make_dir(path):
    '''
    创建目录
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def write_xml(folder: str, img_name: str, path: str, img_width: int, img_height: int, tag_num: int, tag_names: str, box_list:list,save_path:str,url_id:int):
    '''
    VOC标注xml文件生成函数
    :param folder: 文件夹名
    :param img_name:
    :param path:
    :param img_width:
    :param img_height:
    :param tag_num: 图片内的标注框数量
    :param tag_name: 标注名称
    :param box_list: 标注坐标,其数据格式为[[xmin1, ymin1, xmax1, ymax1],[xmin2, ymin2, xmax2, ymax2]....]
    :return: a standard VOC format .xml file, named "img_name.xml"
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
 
    with open(os.path.join(save_path,"{}".format(url_id)+ ".xml"), "w", encoding="utf-8") as f:
        # writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")

def save_img_xml(driver,img_path,xml_path,img,cls_names,box_list,url_id):
    '''
    裁剪一块块的图片，并取出对应的标签
    '''
    img_num = 0
    js_height = "return window.screen.availHeight"
    height = driver.execute_script(js_height)
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
        choosed_coords = []
        choosed_names = []
        for name,coord in zip(cls_names,box_list):
            x1,y1,x2,y2 = coord
            # 坐标重新映射到[0,height]之间
            y1 -= y_start
            y2 -= y_start
            # 只保留当前页范围的元素
            if y2 < 0 or y1 >= height:
                continue
            # 边界裁剪
            if y1 < 0:
                y1 = 0
            if y2 >= height:
                y2 = height-1
            choosed_coords.append([x1,y1,x2,y2])
            choosed_names.append(name)
        img_name = str(url_id) + "_" + str(img_num)
        # 保存图片
        cv2.imwrite(os.path.join(img_path,img_name+".jpg"),save_img)
        # 保存xml
        write_xml(img_path,img_name,img_path,w,height,len(choosed_coords),choosed_names,choosed_coords,xml_path)
        img_num += 1
    print(f"{img_num} imgs saved in {img_path}, xmls saved in {xml_path}")


def checkOverlap(boxa, boxb):
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


def get_box_info(loc, box_lists):
    over_lap = False
    for box in box_lists:
        x1, y1, w1, h1 = loc
        x2, y2, x3, y3 = box
        w2 = x3 - x2
        h2 = y3 - y2
        boxb = (x2,y2,w2,h2)
        overlap = checkOverlap(loc, boxb)
        # print(overlap)
        if overlap > 0.60:
            over_lap = True
            break
    return over_lap
    


def gen_random_anns(driver,img,save_path,url_id,w,h):
    '''
    随机截图并生成标注xml文件
    '''
    # driver.maximize_window()
    width=driver.execute_script("return document.body.clientWidth")
    window_height = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(w,h)
    # window_size = driver.get_window_size()
    # width,window_height = window_size["width"],window_size["height"]
    print("shot: ", width, window_height)
    if width == w and window_height == h:
        
        print("search elements...........")
        t1 = time.time()
        img_num = 0
        driver.implicitly_wait(0.5)
        # 根据tagname查找
        tag_results = get_tag_elements(driver,tag_names)
        # 根据规则查找
        class_results = get_class_contains_elements(driver, search_str)
        results = set(class_results) | set(tag_results)
        
        img_path = os.path.join(save_path,"img")
        xml_path = os.path.join(save_path,"xml")
        # 检查保存路径
        make_dir(save_path)
        make_dir(img_path)
        make_dir(xml_path)
        
        print(f"search elements done, cost {time.time()-t1}")
        print("tag results:",len(tag_results))
        print("class result:",len(class_results))
        t1 = time.time()

        cls_names = []
        box_lists = []

        def get_list(e):
            pbar.update(1)
            # tt = time.time()
            x, y, w, h = int(e.rect['x']), int(e.rect['y']), int(e.rect['width']), int(e.rect['height'])
            loc = (x,y,w,h)
            over_lap = get_box_info(loc, box_lists)
            if not over_lap:
                if e.tag_name!="div":
                    if (w > 8 and h > 8 and [x, y, x + w, y + h] not in box_lists and h < 1500 and w < 1700 and w / h < 30):
                        if not(e.tag_name =="img" and w/h>5):
                            box_lists.append([x,y,x+w,y+h])
                            cls_names.append(e.tag_name)
        
        # 创建多线程处理
        pbar = tqdm(total=len(results))
        pool = ThreadPool(processes=10)
        pool.map(get_list, results)
        pool.close()
        pool.join()
        pbar.close()
        
        print("find cost:",time.time()-t1)
        
        # 保存图片和xml
        #save_img_xml(driver,img_path,xml_path,img,cls_names,box_lists,url_id)
        # cv2.imwrite("./tmp3/img"+"/{}.png".format(url_id),img)
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(img_path,str(url_id)+".jpg"))
        write_xml(img_path, str(img_num), img_path, width, window_height, len(box_lists), cls_names, box_lists, xml_path,url_id)

def init_driver():
    '''
    初始化webdriver
    '''
    chrome_options = Options()
    
    chrome_options.add_argument('--headless')                                           # 添加无头参数r,一定要使用无头模式，不然截不了全页面，只能截到你电脑的高度
    chrome_options.add_argument('--disable-gpu')                                        # 为了解决一些莫名其妙的问题关闭 GPU 计算
    chrome_options.add_argument('--no-sandbox')                                         # 为了解决一些莫名其妙的问题浏览器不动
    chrome_options.add_argument('--log-level=3')                                        # 关闭无效的警告打印信息
    driver_width, driver_height = pyautogui.size()                                      # 通过pyautogui方法获得屏幕尺寸
    chrome_options.add_argument('--window-size=%sx%s' % (driver_width, driver_height))  # 设置浏览器窗口大小
    driver = webdriver.Chrome(options=chrome_options)
    
    return driver 

def change_address(postal):
    '''
    将右边切换到纽约10041
    '''
    # while True:
    try:
        # driver.find_element_by_id('glow-ingress-line1').click()
        driver.find_element(By.XPATH,"//*[@id='nav-main']/div[1]/div/div/div[3]/span[2]/span/input").click()
        # driver.find_element_by_id('nav-global-location-slot').click()
        time.sleep(2)
    except Exception as e:
        driver.refresh()
        time.sleep(10)
        # continue
    try:
        driver.find_element(By.XPATH,"//*[@id='GLUXZipUpdateInput']").send_keys(postal)
        time.sleep(1)
    except Exception :
        driver.refresh()
        time.sleep(10)
        # continue
    
    try:
        driver.find_element(By.XPATH,"//*[@id='GLUXZipUpdate']/span/input").click()
        time.sleep(1)
        # break
    except Exception :
        driver.refresh()
        time.sleep(10)
        # continue
    driver.refresh()
    time.sleep(1)


def save_screen_to_png(driver):
    '''
    截图全屏图片
    '''
    driver.implicitly_wait(10)
    
    # 模拟人滚动滚动条,处理图片懒加载问题
    js_height = "return document.body.clientHeight"

    k = 1
    height = driver.execute_script(js_height)
    try:
        
        while True:
            if k * 500 < height:
                js_move = "window.scrollTo(0,{})".format(k * 500)
                # print(js_move)
                driver.execute_script(js_move)
                time.sleep(1)
                height = driver.execute_script(js_height)
                k += 1
            else:
                break
        width=driver.execute_script("return document.body.clientWidth")
        height = driver.execute_script("return document.documentElement.scrollHeight")
        driver.set_window_size(width,height)
        print("shot: ", width, height)
        img_bin = driver.get_screenshot_as_png()
        image = np.asarray(bytearray(img_bin), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return img, width, height
        
    except Exception as e:
        print(e)
        return None

def get_screen_full(driver):
    '''
    从上到下截全屏，问题是截的不够全
    '''
    driver.implicitly_wait(10)
    time.sleep(10)
    # 全屏截图的关键，用js获取页面的宽高
    width=driver.execute_script("return document.body.clientWidth")
    height=driver.execute_script("return document.body.scrollHeight")
    print(width,height)

    # width, height = pyautogui.size()   
    # print(width,height)
    # 获取浏览器的宽高
    driver.set_window_size(width,height)
    # 截图base64
    img_bin = driver.get_screenshot_as_png()
    image = np.asarray(bytearray(img_bin), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return img



def get_data_from_url(driver,url,url_id):
    driver.get(url)
    # 修改邮编
    time.sleep(10)
    driver.implicitly_wait(10)
    post_id = 10041
    try:
        change_address(post_id)
    except Exception as e:   
        print(e)
    t1 = time.time()
    print("load page and change address cost:",t1-st)

    # 保存全屏截图
    

    img, w, h = save_screen_to_png(driver)
    
    
    t2 = time.time()
    print("save screen cost:",t2-t1)

   
    save_path = "F:/Datasets/UIED/元素检测/原始标注数据/2023_02_17/cunmin2"
    
    # 查找元素

    gen_random_anns(driver,img,save_path,url_id,w,h)

import json
if __name__ == "__main__":

    # 初始化webDriver
    

    st = time.time()
    url = "https://www.amazon.com/"
    url = "https://www.amazon.com/CUPSHE-Casual-Summer-Crochet-Dresses/dp/B0BTSV3187/ref=sr_1_2?content-id=amzn1.sym.b24fa8ec-eb31-46d1-a5f8-fe8bcdc3d018%3Aamzn1.sym.b24fa8ec-eb31-46d1-a5f8-fe8bcdc3d018&pd_rd_r=3c7482f3-7950-4e95-965a-5c6f765cf2a1&pd_rd_w=Zd6XU&pd_rd_wg=CNW0m&pf_rd_p=b24fa8ec-eb31-46d1-a5f8-fe8bcdc3d018&pf_rd_r=R8GGX3G1DBHBK036NCT5&qid=1675762746&s=apparel&sr=1-2&wi=lbfp6fbf_0"
    url = "https://www.amazon.com/UGG-Scuffette-Slipper-Chestnut-Size/dp/B082HJ2NQN/ref=sr_1_3?isTryState=0&nodeID=14807110011&pd_rd_r=ed856e00-e5ac-4ed1-8537-34fcdff755e9&pd_rd_w=n92qA&pd_rd_wg=KQZmf&pf_rd_p=72d0c0b8-8a33-49dd-8a98-91f9fbc2fe19&pf_rd_r=65VDNKKWAZ44HEM36PNW&psd=1&qid=1675838043&refinements=p_n_feature_eighteen_browse-bin%3A21451213011&s=prime-wardrobe&sr=1-3&th=1"
    url = "https://www.amazon.com/UGG-Ansley-Slipper-Black-Size/dp/B082HJ9H4S/ref=d_softlines_sb_mfpfy_btf_v1_vft_none_sccl_1_2/139-7571617-7444346?pd_rd_w=ze2Pc&content-id=amzn1.sym.6a7ee8bc-3980-4d7b-9042-d97e0c49e955&pf_rd_p=6a7ee8bc-3980-4d7b-9042-d97e0c49e955&pf_rd_r=GX9PWY3YCHGG7W2T1SD6&pd_rd_wg=XsYg0&pd_rd_r=dc9eb53e-6c30-47f5-a836-ab83ab225f03&pd_rd_i=B0BGM39FXG&psc=1"
    tag_names = ["button",  # 按钮
                "img",      # 图片
                # "i",        # ico图标
                # "svg",      # svg格式图标
                # "use",      # SVG图标的节点获取
                "input",    # 输入框
                "table",    # 表格
                "select",   # 下拉框
                ] 
        
    search_str = ["title", 
                  "btn",
                  "button", 
                  "arrow", 
                  "select", 
                  "ico", 
                  "img", 
                  'logo', ]
    
    tag_names = ["button",  # 按钮
            "img",      # 图片
            # "i",        # ico图标
            # "svg",      # svg格式图标
            # "use",      # SVG图标的节点获取
            "input",    # 输入框
            # "span",     # 带背景的区域
            # "em",       # 文本定义为强调内容
            # "table",    # 表格
            "select"
        
            ]  # 下拉框
        
    search_str = ['nav_a','nav-a','nav-a-content','a-button-inner',
                'a-button-text', "a-expander-prompt", 'a-icon',"a-input-text", 'a-declarative', 'a-meter',"a-price-whole"
                'cr-lighthouse-term', "cr-helpful-text", "action-inner", "sign-in-tooltip-link","s-pagination-item", "nav-search-scope", 
                "nav-hamburger-menu", "a-spacing-micro", "play-button-inner","a-size-base-plus", "icp-button", "padding-left-small", "a-link-normal","nav-menu-item", "nav-menu-cta", "pui-text"]
    for i in range(1000):
        driver = init_driver()
        with open("urls.json", 'r') as f:
            url_data = json.load(f)
            rand_int = random.randint(1, len(url_data))
            print(rand_int)
            rand_int =rand_int
            url = url_data[rand_int]
            print(url)
            try:
                get_data_from_url(driver, url, rand_int)
                driver.close()
            except Exception as e:
                print(e)
    # et = time.time()
    # print("gen random anns cost:",et-t2)
    # print("total cost:",et-st)
