# -*- coding:utf-8 -*-
# author: DuanshengLiu
import cv2
import numpy as np

def findVertices(points):
    # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
    rect = cv2.minAreaRect(points)
    # 获取矩形四个顶点，浮点型
    box = cv2.boxPoints(rect).astype(np.int32)
    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    # left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    # right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    # top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    # bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]

    left_point_y = np.max(box[:, 1][np.where(box[:, 0] == left_point_x)])
    right_point_y = np.min(box[:, 1][np.where(box[:, 0] == right_point_x)])
    top_point_x = np.min(box[:, 0][np.where(box[:, 1] == top_point_y)])
    bottom_point_x = np.max(box[:, 0][np.where(box[:, 1] == bottom_point_y)])


    # 上下左右四个点坐标
    vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [right_point_x, right_point_y]])
    return vertices, rect

def tiltCorrection(vertices, rect):
    # 畸变情况1
    if rect[2] > -45:
        new_right_point_x = vertices[0, 0]
        new_right_point_y = int(vertices[1, 1] - (vertices[0, 0]- vertices[1, 0]) / (vertices[3, 0] - vertices[1, 0]) * (vertices[1, 1] - vertices[3, 1]))
        new_left_point_x = vertices[1, 0]
        new_left_point_y = int(vertices[0, 1] + (vertices[0, 0] - vertices[1, 0]) / (vertices[0, 0] - vertices[2, 0]) * (vertices[2, 1] - vertices[0, 1]))
        # # 校正后的四个顶点坐标
        point_set_1 = np.float32([[440, 0],[0, 0],[0, 140],[440, 140]])
    # 畸变情况2
    elif rect[2] < -45:
        new_right_point_x = vertices[1, 0]
        new_right_point_y = int(vertices[0, 1] + (vertices[1, 0] - vertices[0, 0]) / (vertices[3, 0] - vertices[0, 0]) * (vertices[3, 1] - vertices[0, 1]))
        new_left_point_x = vertices[0, 0]
        new_left_point_y = int(vertices[1, 1] - (vertices[1, 0] - vertices[0, 0]) / (vertices[1, 0] - vertices[2, 0]) * (vertices[1, 1] - vertices[2, 1]))
        # # 校正后的四个顶点坐标
        point_set_1 = np.float32([[0, 0],[0, 140],[440, 140],[440, 0]])

        # 校正前平行四边形四个顶点坐标
    new_box = np.array([(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]), (new_right_point_x, new_right_point_y)])
    point_set_0 = np.float32(new_box)
    return point_set_0, point_set_1, new_box


def point_to_line_distance(X, Y, x0, y0, x2, y2):
    if x2 - x0:
        k_up = (y2 - y0) / (x2 - x0)  # 斜率不为无穷大
        d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
    else:  # 斜率无穷大
        d_up = abs(X - x2)

    return d_up

def locate_and_correct(img_src, img_mask):
    """
    该函数通过cv2对img_mask进行边缘检测，获取车牌区域的边缘坐标(存储在contours中)和最小外接矩形4个端点坐标,
    再从车牌的边缘坐标中计算出和最小外接矩形4个端点最近的点即为平行四边形车牌的四个端点,从而实现车牌的定位和矫正
    :param img_src: 原始图片
    :param img_mask: 通过u_net进行图像分隔得到的二值化图片，车牌区域呈现白色，背景区域为黑色
    :return: 定位且矫正后的车牌
    """
    # cv2.imshow('img_mask',img_mask)
    # cv2.waitKey(0)
    ret,thresh = cv2.threshold(img_mask[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #二值化
    # cv2.imshow('thresh',thresh)
    img_mask = thresh
    # cv2.imshow('img_mask',img_mask)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(img_mask[:, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):  # contours1长度为0说明未检
        # 测到车牌
        print("未检测到车牌")
        return [], []
    else:
        Lic_img = []
        img_src_copy = img_src.copy()  # img_src_copy用于绘制出定位的车牌轮廓
        for ii, cont in enumerate(contours):

            perimeter = cv2.arcLength(cont, True)
            epsilon = 0.05*cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)


            cv2.drawContours(img_mask, approx, -1, (0, 0, 255), 3)
            # cv2.imshow('img_mask',img_mask)
            # cv2.waitKey(0)

            x, y, w, h = cv2.boundingRect(cont)  # 获取最小外接矩形
            img_cut_mask = img_mask[y:y + h, x:x + w]  # 将标签车牌区域截取出来
            # cv2.imshow('img_cut_mask',img_cut_mask)
            # cv2.waitKey(0)
            print('w,h,均值,宽高比',w,h,np.mean(img_cut_mask),w/h)
            # contours中除了车牌区域可能会有宽或高都是1或者2这样的小噪点，
            # 而待选车牌区域的均值应较高，且宽和高不会非常小，因此通过以下条件进行筛选
            if np.mean(img_cut_mask) >= 75 and w > 15 and h > 15:

                # x_min = np.min(cont[ :, :, 0])
                # x_max = np.max(cont[ :, :, 0])
                # y_min = np.min(cont[ :, :, 1])
                # y_max = np.max(cont[ :, :, 1])

                # 获取最可疑区域轮廓点集
                # points = np.array(cont[:, 0])
                #
                # vertices, rect = findVertices(cont)
                #
                # cv2.drawContours(img_mask, vertices, 0, (0, 255, 0), 2)
                # cv2.imshow('img_mask',img_mask)
                #
                #
                # point_set_0, new_box = tiltCorrection(vertices, rect)


                rect = cv2.minAreaRect(cont)  # 针对坐标点获取带方向角的最小外接矩形，中心点坐标，宽高，旋转角度
                box = cv2.boxPoints(rect).astype(np.int32)  # 获取最小外接矩形四个顶点坐标
                # # cv2.drawContours(img_mask, contours, -1, (0, 0, 255), 2)
                # # cv2.drawContours(img_mask, [box], 0, (0, 255, 0), 2)
                # # cv2.imshow('img_mask',img_mask)
                # # cv2.waitKey(0)
                # cont = cont.reshape(-1, 2).tolist()

                approx = approx.reshape(-1, 2).tolist()

            # # 由于转换矩阵的两组坐标位置需要一一对应，因此需要将最小外接矩形的坐标进行排序，最终排序为[左上，左下，右上，右下]
                box = sorted(box, key=lambda xy: xy[0])  # 先按照左右进行排序，分为左侧的坐标和右侧的坐标
                box_left, box_right = box[:2], box[2:]  # 此时box的前2个是左侧的坐标，后2个是右侧的坐标
                box_left = sorted(box_left, key=lambda x: x[1])  # 再按照上下即y进行排序，此时box_left中为左上和左下两个端点坐标
                box_right = sorted(box_right, key=lambda x: x[1])  # 此时box_right中为右上和右下两个端点坐标
                box = np.array(box_left + box_right)  # [左上，左下，右上，右下]
                # print(box)
                x0, y0 = box[0][0], box[0][1]  # 这里的4个坐标即为最小外接矩形的四个坐标，接下来需获取平行(或不规则)四边形的坐标
                x1, y1 = box[1][0], box[1][1]
                x2, y2 = box[2][0], box[2][1]
                x3, y3 = box[3][0], box[3][1]
                #
                # # def point_to_line_distance(X, Y):
                # #     if x2 - x0:
                # #         k_up = (y2 - y0) / (x2 - x0)  # 斜率不为无穷大
                # #         d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
                # #     else:  # 斜率无穷大
                # #         d_up = abs(X - x2)
                # #     if x1 - x3:
                # #         k_down = (y1 - y3) / (x1 - x3)  # 斜率不为无穷大
                # #         d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
                # #     else:  # 斜率无穷大
                # #         d_down = abs(X - x1)
                # #     return d_up, d_down
                #
                #
                # d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
                l0, l1, l2, l3 = [x0, y0], [x1, y1], [x2, y2], [x3, y3]
                # for each in cont:  # 计算cont中的坐标与矩形四个坐标的距离以及到上下两条直线的距离，对距离和进行权重的添加，成功计算选出四边形的4个顶点坐标
                #     x, y = each[0], each[1]
                #     dis0 = (x - x0) ** 2 + (y - y0) ** 2
                #     dis1 = (x - x1) ** 2 + (y - y1) ** 2
                #     dis2 = (x - x2) ** 2 + (y - y2) ** 2
                #     dis3 = (x - x3) ** 2 + (y - y3) ** 2
                #     # d_up, d_down = point_to_line_distance(x, y)
                #     d_up = point_to_line_distance(x, y, x0, y0, x2, y2)
                #     d_down = point_to_line_distance(x, y, x1, y1, x3, y3)
                #
                #
                #     weight = 0.5
                #     if weight * d_up + (1 - weight) * dis0 < d0:  # 小于则更新
                #         d0 = weight * d_up + (1 - weight) * dis0
                #         l0 = (x, y)
                #     if weight * d_down + (1 - weight) * dis1 < d1:
                #         d1 = weight * d_down + (1 - weight) * dis1
                #         l1 = (x, y)
                #     if weight * d_up + (1 - weight) * dis2 < d2:
                #         d2 = weight * d_up + (1 - weight) * dis2
                #         l2 = (x, y)
                #     if weight * d_down + (1 - weight) * dis3 < d3:
                #         d3 = weight * d_down + (1 - weight) * dis3
                #         l3 = (x, y)
                # 左上角，右下角, 左下角，右上角，
                # l0, l1, l2, l3 = point_set_0[0], point_set_0[1], point_set_0[2], point_set_0[3]
                # l0, l1, l2, l3 = approx[0], approx[1], approx[2], approx[3]
                print([l0,l1,l2,l3])
                for l in [l0, l1, l2, l3]:
                    cv2.circle(img=img_mask, color=(255, 0, 0), center=tuple(l), thickness=2, radius=5)
                cv2.imshow('img_mask',img_mask)

                p0, p1, _ = tiltCorrection([l0,l1,l2,l3], rect)

                # # cv2.waitKey(0)
                # p0 = np.float32([l0, l1, l3, l2])  #，p0和p1中的坐标顺序对应，以进行转换矩阵的形成
                # p1 = np.float32([(0, 0), (0, 80), (240, 80), (240, 0)])  # 我们所需的长方形
                transform_mat = cv2.getPerspectiveTransform(p0, p1)  # 构成转换矩阵
                lic = cv2.warpPerspective(img_src, transform_mat, (440, 140))  # 进行车牌矫正
                # cv2.imshow('lic',lic)
                # cv2.waitKey(0)
                Lic_img.append(lic)
                cv2.drawContours(img_src_copy, [np.array([l0, l1, l3, l2])], -1, (0, 255, 0), 2)  # 在img_src_copy上绘制出定位的车牌轮廓，(0, 255, 0)表示绘制线条为绿色
    return img_src_copy, Lic_img
