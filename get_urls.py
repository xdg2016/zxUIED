
import time
from selenium import webdriver
import json
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
url = "https://www.amazon.com/ref=nav_logo"

driver = webdriver.Chrome()
def get_urls(url):
    
    url_lists = []
    driver.maximize_window()
    driver.implicitly_wait(6)
    
    driver.get(url)
    time.sleep(3)
    post_id = 10041
    change_address(post_id)
    for link in driver.find_elements_by_xpath("//*[@href]"):
        url = link.get_attribute('href')
        if "amazon" in url:
            url_lists.append(url)
    return url_lists


def change_address(postal):

    try:
        # driver.find_element_by_id('glow-ingress-line1').click()
        driver.find_element(By.XPATH,"//*[@id='nav-main']/div[1]/div/div/div[3]/span[2]/span/input").click()
        # driver.find_element_by_id('nav-global-location-slot').click()
        time.sleep(2)
    except Exception as e:
        driver.refresh()
        time.sleep(10)

    try:
        driver.find_element(By.XPATH,"//*[@id='GLUXZipUpdateInput']").send_keys(postal)
        time.sleep(1)
    except Exception :
        driver.refresh()
        time.sleep(10)

    
    try:
        driver.find_element(By.XPATH,"//*[@id='GLUXZipUpdate']/span/input").click()
        time.sleep(1)
    except Exception :
        driver.refresh()
        time.sleep(10)
     
    driver.refresh()
    time.sleep(1)


all_url_list = []

url_list = get_urls(url)
print(len(url_list))
for item in url_list:
    urls = get_urls(item)
    print(len(urls))
    if(len(url)>15):
        all_url_list += urls
        all_url_list = list(set(all_url_list))


        with open("urls.json", 'w') as f:
            json.dump(all_url_list,f)
    



# print(url)
# print(len(url_list))