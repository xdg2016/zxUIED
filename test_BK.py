from bs4 import BeautifulSoup
from pprint import pprint
import pyautogui
#引入selenium库中的 webdriver 模块
from selenium import webdriver
from selenium.webdriver.common.by import By
from multiprocessing.dummy import Pool as ThreadPool
from selenium.webdriver.chrome.options import Options
#引入time库
import time
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
import random

from PIL import Image

def get_tag_elements(driver,tag_names):
    '''
    根据html标签名查找
    '''
    all_results = []
    for tag in tag_names:
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
    for cls in classes:
        results = driver.find_elements(By.XPATH, f"//*[starts-with(@class,'{cls}')]")
        # results =  driver.find_elements(By.XPATH, f"//*[class='{cls}']")
        all_results.extend(results)
    return all_results

# def draw(img,results,save_path = "anna.png"):
#     '''
#     将解析的结果画在图上
#     '''
#     for e in results:
#         try:
#             # print(e.rect)
#             x,y,w,h = int(e.rect['x']),int(e.rect['y']),int(e.rect['width']),int(e.rect['height'])
#             pt1 = x,y
#             pt2 = x+w,y+h
#             cv2.rectangle(img,pt1,pt2,color=(0,0,255),thickness=2)
#             cv2.putText(img, e.tag_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
#         except:
#             continue

#     # cv2.imshow("img",img_)
#     cv2.imwrite(save_path,img)
#     cv2.waitKey(0)

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
    if not os.path.exists(path):
        os.makedirs(path)


cls_names = []
box_lists = []

def get_box_cls(e):
    x,y,w,h = int(e.rect['x']),int(e.rect['y']),int(e.rect['width']),int(e.rect['height'])
    if w<1000 and h<800:
        box_lists.append([x,y,x+w,y+h])
        cls_names.append(e.tag_name)

def gen_random_anns(driver,times,save_path):
    '''
    随机截图并生成标注xml文件
    '''
    # driver.maximize_window()
    width=driver.execute_script("return document.body.clientWidth")
    window_height = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(width,window_height)
    # window_size = driver.get_window_size()
    # width,window_height = window_size["width"],window_size["height"]
    print("shot: ", width, window_height)
    img_num = 0
    # elements = driver.find_elements(By.XPATH,f"//*")
    tag_results = get_tag_elements(driver,tag_names)
    # 根据规则查找
    results = get_class_contains_elements(driver, search_str)
    results = set(results) | set(tag_results)
    
    img_path = os.path.join(save_path,"img")
    xml_path = os.path.join(save_path,"xml")
    
    
    use_mp = True
    if use_mp:
        datas = [data for data in results]
        pool = ThreadPool(processes = 10)
        pool.map(get_box_cls, datas)
    else:
        for e in results:
            try:
        
                x,y,w,h = int(e.rect['x']),int(e.rect['y']),int(e.rect['width']),int(e.rect['height'])

                # print("cost :",t2-t1)
                box_lists.append([x,y,x+w,y+h])
                cls_names.append(e.tag_name)
            except:
                continue
    print("len(box_lists): ",len(box_lists))
    make_dir(save_path)
    make_dir(img_path)
    make_dir(xml_path)
    
    # img = get_screen_full(driver)
    # cv2.imwrite(img_path+"/0.png",img)        
    write_xml(img_path,str(img_num),img_path,width,window_height,len(box_lists),cls_names,box_lists,xml_path)
    img_num += 1
    print(f"saved imgs {img_num}")



def init_driver():

    #打开谷歌浏览器
    chrome_options = Options()
    # 2> 添加无头参数r,一定要使用无头模式，不然截不了全页面，只能截到你电脑的高度
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--log-level=3')                                        # 关闭打印selenium信息
    # 3> 为了解决一些莫名其妙的问题关闭 GPU 计算
    chrome_options.add_argument('--disable-gpu')
    # 4> 为了解决一些莫名其妙的问题浏览器不动
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('headless')                                             # 无头模式下才能截长图
    driver_width, driver_height = pyautogui.size()  # 通过pyautogui方法获得屏幕尺寸
    print(driver_width, driver_height)
    chrome_options.add_argument('--window-size=%sx%s' % (driver_width, driver_height))  # 设置浏览器窗口大小
    driver = webdriver.Chrome(options=chrome_options)
    

    return driver 

def change_address(postal):
    while True:
        try:
            # driver.find_element_by_id('glow-ingress-line1').click()
            driver.find_element(By.XPATH,"//*[@id='nav-main']/div[1]/div/div/div[3]/span[2]/span/input").click()
            # driver.find_element_by_id('nav-global-location-slot').click()
            time.sleep(2)
        except Exception as e:
            driver.refresh()
            time.sleep(10)
            continue
        try:
            driver.find_element(By.XPATH,"//*[@id='GLUXZipUpdateInput']").send_keys(postal)
            time.sleep(1)
        except Exception :
            driver.refresh()
            time.sleep(10)
            continue
        
        try:
            driver.find_element(By.XPATH,"//*[@id='GLUXZipUpdate']/span/input").click()
            time.sleep(1)
            break
        except Exception :
            driver.refresh()
            time.sleep(10)
            continue
    driver.refresh()
    time.sleep(1)



def save_screen_to_png(driver):
    driver.implicitly_wait(10)

    # driver.maximize_window()
    # width=driver.execute_script("return document.body.clientWidth")
    # height=driver.execute_script("return document.documentElement.scrollHeight")
    

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
 
        time.sleep(1)
 
        # 7>  # 直接截图截不全，调取最大网页截图
        
        # # width, height = pyautogui.size()   
        
        # width, height = pyautogui.size()   
        # print(width,height)
        # # 获取浏览器的宽高
        # driver.set_window_size(width, height)
        width=driver.execute_script("return document.body.clientWidth")
        height = driver.execute_script("return document.documentElement.scrollHeight")
        driver.set_window_size(width,height)
        print("shot: ", width, height)
        # 获取浏览器的宽高
        
        
        
        time.sleep(1)
        # 截图并关掉浏览器
        driver.save_screenshot("D:/workspace/zxUIED/zxUIED/tmp2/img/0.png")
        # driver.close()

 
    except Exception as e:
        print(e)


def get_screen_full(driver):
    driver.implicitly_wait(20)
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


if __name__ == "__main__":

    # 初始化webDriver
    driver = init_driver()


    url = "https://www.amazon.com/"
    url = "https://www.amazon.com/b?node=16225016011&pf_rd_r=QWCTYPD7HS3RK4825EK4&pf_rd_p=e5b0c85f-569c-4c90-a58f-0c0a260e45a0&pd_rd_r=a111a7b9-1c9a-4f1c-94cd-adcebf80edc8&pd_rd_w=91i37&pd_rd_wg=maykA&ref_=pd_gw_unk"
    url = "https://www.amazon.com/CUPSHE-Casual-Summer-Crochet-Dresses/dp/B0BTSV3187/ref=sr_1_2?content-id=amzn1.sym.b24fa8ec-eb31-46d1-a5f8-fe8bcdc3d018%3Aamzn1.sym.b24fa8ec-eb31-46d1-a5f8-fe8bcdc3d018&pd_rd_r=3c7482f3-7950-4e95-965a-5c6f765cf2a1&pd_rd_w=Zd6XU&pd_rd_wg=CNW0m&pf_rd_p=b24fa8ec-eb31-46d1-a5f8-fe8bcdc3d018&pf_rd_r=R8GGX3G1DBHBK036NCT5&qid=1675762746&s=apparel&sr=1-2&wi=lbfp6fbf_0"
    url = "https://www.amazon.com/UGG-Scuffette-Slipper-Chestnut-Size/dp/B082HJ2NQN/ref=sr_1_3?isTryState=0&nodeID=14807110011&pd_rd_r=ed856e00-e5ac-4ed1-8537-34fcdff755e9&pd_rd_w=n92qA&pd_rd_wg=KQZmf&pf_rd_p=72d0c0b8-8a33-49dd-8a98-91f9fbc2fe19&pf_rd_r=65VDNKKWAZ44HEM36PNW&psd=1&qid=1675838043&refinements=p_n_feature_eighteen_browse-bin%3A21451213011&s=prime-wardrobe&sr=1-3&th=1"
    url = "https://www.amazon.com/home-garden-kitchen-furniture-bedding/b/?ie=UTF8&node=1055398&ref_=nav_cs_home"
    url = "https://www.amazon.com/b?node=3736181&ref=sv_hg_fl_3736181"
    driver.get(url)
    # 修改邮编
    time.sleep(5)
    post_id = 10041
    st = time.time()
    change_address(post_id)
    print("change_address cost time:{}".format(time.time()-st))
    st = time.time()
    save_screen_to_png(driver)
    print("save_screen_to_png cost time:{}".format(time.time()-st))
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
        
    search_str = ["select", 'nav_a','nav-a','nav-a-content', 'a-button-inner',
   'a-button-text', "a-expander-prompt", 'a-icon',"a-input-text", 'a-declarative', 'a-meter',
    'cr-lighthouse-term', "cr-helpful-text", "action-inner", "sign-in-tooltip-link", "nav-search-scope", "nav-hamburger-menu", "a-spacing-micro","icp-button"]

    save_path = "D:/workspace/zxUIED/zxUIED/tmp2"
    st = time.time()
    gen_random_anns(driver, 10, save_path)
    print("gen_random_anns cost time:{}".format(time.time()-st))