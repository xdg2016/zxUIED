# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import cv2
import numpy as np
import onnxruntime as ort

class ResNet():
    def __init__(self,
                 model_pb_path,
                 label_path,):
        # 读取类别文件，获取类别列表
        self.classes = {}
        with open(label_path, 'r',encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                id,name = line.split()
                self.classes[id] = name

        self.num_classes = len(self.classes)
       
        # 均值、标准差，用于归一化（BGR顺序）
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        # 初始化onnx推理
        self.net = self.onnx_init(model_pb_path)      
        # 根据网络结构，获取输入名称和尺寸
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net.get_inputs())
        }
        self.input_shape = inputs_shape['x'][2:]


    def onnx_init(self,model_path):
        '''
        onnx模型初始化
        '''
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            net = ort.InferenceSession(model_path, so)
        except Exception as ex:
            print(ex)
            net = None
        return net

    def _normalize(self, img):
        '''
        图像归一化
        '''
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        '''
        图像缩放

        Args:
            srcimg 原始输入图片
        Returns:
            keep_ratio 是否保持原图宽高比
        '''
        origin_shape = srcimg.shape[:2]
        neww,newh = det_w,det_h
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        img_shape = np.array([
            [float(newh), float(neww)]
        ]).astype('float32')
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
        img = cv2.resize(srcimg,  (neww,newh), interpolation=2)

        return img, img_shape, scale_factor

    def preprocess(self,srcimg):
        '''
        数据预处理
        '''
        
        # 缩放到推理尺寸
        img, im_shape, scale_factor = self.resize_image(srcimg)
        # 按照BGR做做归一化
        img = self._normalize(img)
        # 再转成RGB
        img = img[:,:,::-1]
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        inputs_dict = {
            'x': blob,
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        return net_inputs
    
    def cls_onnx(self, srcimg):
        '''
        目标检测模型推理接口

        Args:
            srcimg 原始数据
        Returns:
            result_list 检测结果列表
        '''
        net_inputs = self.preprocess(srcimg)
        defualt_label = "-1"
        try:
            ts = time.time()
            t = 1
            for i in range(t):
                outs = self.net.run(None, net_inputs)
            # print("infer cost:",(time.time()-ts)/t)

            label = np.argmax(outs[0][0])      
            if label != defualt_label:
                result = {
                    "id" : label,
                    "label": self.classes[str(label)]
                }

        except Exception as e:
            print(e)
            result = {
                    "id" : defualt_label,
                    "label": "null"
                }

        return result

# 默认推理尺寸
det_h = 224
det_w = 224