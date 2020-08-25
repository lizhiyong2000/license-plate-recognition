from algorithm.api import LprAlgorithm

if __name__ == '__main__':
    lpr = LprAlgorithm()
    img_src = "../images/download/新能源车牌/Baidu_0001.jpeg"

    image, mask, lic = lpr.detect(img_src)

    print(image)
    print(mask)
    print(lic)