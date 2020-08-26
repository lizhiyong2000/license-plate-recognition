import os

import cv2
import numpy as np

from algorithm.api import LprAlgorithm

#将json文件label转换为到label文件夹
def json_to_image(path):
    # path = '../images/download/新能源车牌/'
    image_dir = os.path.join(path, 'image')
    json_dir = os.path.join(path, 'json')
    label_dir = os.path.join(path, 'label')

    # if not os.path.exists(train_image):
    #     os.makedirs(train_image)

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    included_extensions = ['json']
    file_names = [fn for fn in os.listdir(json_dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    file_names.sort()
    for i in range(len(file_names)):
        file_name = file_names[i]
        file_name_prefix = file_name[:len(file_name) - 5]
        json_path = os.path.join(json_dir, file_name)
        dataset_path = os.path.join(json_dir, file_name_prefix)
        print(dataset_path)
        cmd = 'labelme_json_to_dataset %s -o %s' % (json_path, dataset_path)
        os.system(cmd)

        # img=cv2.imread('D:/desktop/labelme/data/%d_json/img.png'%i)
        label = cv2.imread(os.path.join(dataset_path, 'label.png'))
        # print(img.shape)
        label=label/np.max(label[:,:,2])*255
        label[:,:,0]=label[:,:,1]=label[:,:,2]
        print(np.max(label[:,:,2]))
        # cv2.imshow('l',label)
        # cv2.waitKey(0)
        # print(set(label.ravel()))
        # cv2.imwrite(train_image+'%d.png'%i,img)
        cv2.imwrite(os.path.join(label_dir, '%s.png' % file_name_prefix), label)



def resize_image(path):

    # path = '../images/download/新能源车牌/'

    included_extensions = ['jpg', 'jpeg', 'png']
    file_names = [fn for fn in os.listdir(path)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    file_names.sort()


    # input_name = os.listdir(path)

    n = len(file_names)
    print(n)
    for i in range(n):
        print("正在读取第%d张图片" % i)

        origin_filepath = os.path.join(path, file_names[i])
        print("图片路径:%s" % origin_filepath )

        img_data = np.fromfile(origin_filepath, dtype=np.uint8)
        img_src = cv2.imdecode(img_data, -1)  # 从中文路径读取时用

        img_resized = cv2.resize(img_src, dsize=(512, 512), interpolation=cv2.INTER_AREA)[:, :, :3]
        filename = os.path.join(path, '%d.png' % (i+1))
        cv2.imwrite(filename, img_resized)

        os.remove(origin_filepath)

def resize_image_replace(path):

    # path = '../images/download/新能源车牌/'

    included_extensions = ['jpg', 'jpeg', 'png']
    file_names = [fn for fn in os.listdir(path)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    file_names.sort()


    # input_name = os.listdir(path)

    n = len(file_names)
    print(n)
    for i in range(n):
        print("正在读取第%d张图片" % i)

        origin_filepath = os.path.join(path, file_names[i])
        print("图片路径:%s" % origin_filepath )

        img_data = np.fromfile(origin_filepath, dtype=np.uint8)
        img_src = cv2.imdecode(img_data, -1)  # 从中文路径读取时用

        img_resized = cv2.resize(img_src, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)[:, :, :3]
        # filename = os.path.join(path, '%d.png' % (i+1))
        cv2.imwrite(origin_filepath, img_resized)

        # os.remove(origin_filepath)

if __name__ == '__main__'
    pass
    # lpr = LprAlgorithm()
    # img_src = "../images/download/新能源车牌/Baidu_0001.jpeg"
    #
    # image, mask, lic = lpr.detect(img_src)
    #
    # print(image)
    # print(mask)
    # print(lic)
    #
    # resize_image('../images/download/货车黄牌/')
    # resize_image('../images/unet_dataset/train/车牌/image')
    # resize_image('../images/unet_dataset/train/新能源车牌/image')
    # resize_image('../images/unet_dataset/train/货车黄牌/image')

    # json_to_image('../images/unet_dataset/train/新能源车牌/')
    # resize_image_replace('../images/unet_dataset/train/新能源车牌/image')
    # resize_image_replace('../images/unet_dataset/train/新能源车牌/label')

    # resize_image_replace('../images/unet_dataset/train/车牌/image')
    # resize_image_replace('../images/unet_dataset/train/新能源车牌/image')
    # resize_image_replace('../images/unet_dataset/train/货车黄牌/image')