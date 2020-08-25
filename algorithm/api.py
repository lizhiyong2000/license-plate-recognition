import cv2
import numpy as np
from tensorflow import keras

from algorithm.CNN import cnn_predict
from algorithm.Unet import unet_predict
from algorithm.core import locate_and_correct


class LprAlgorithm:
    def __init__(self):
        self.unet = keras.models.load_model('unet.h5')
        self.cnn = keras.models.load_model('cnn.h5')
        print('正在启动中,请稍等...')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])
        print("已启动,开始识别吧！")

    def detect(self, img_src_path):
        img_data = np.fromfile(img_src_path, dtype=np.uint8)
        img_src = cv2.imdecode(img_data, -1)  # 从中文路径读取时用
        h, w = img_src.shape[0], img_src.shape[1]
        if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
            return img_src, None, None
        else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
            img_src, img_mask = unet_predict(self.unet, img_src_path)
            img_src_copy, lic_img = locate_and_correct(img_src, img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
            return img_src_copy, img_mask, lic_img

    def detect_and_predict(self, img_src_path):
        img_data = np.fromfile(img_src_path, dtype=np.uint8)
        img_src = cv2.imdecode(img_data, -1)  # 从中文路径读取时用
        h, w = img_src.shape[0], img_src.shape[1]
        if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
            lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
            img_src_copy, lic_img = img_src, [lic]
        else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
            img_src, img_mask = unet_predict(self.unet, img_src_path)
            img_src_copy, lic_img = locate_and_correct(img_src, img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正

        lic_pred = cnn_predict(self.cnn, lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
        return lic_pred
