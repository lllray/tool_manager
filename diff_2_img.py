import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_images(image1, image2):
    # 读取两个图像
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否成功读取
    if img1 is None or img2 is None:
        print("无法读取图像")
        return

    # 比较两个图像并作差
    diff = cv2.absdiff(img1, img2) * 200

    height, width = img1.shape[:2]

    img1 = cv2.resize(img1, (height * 5, width * 5))
    img2 = cv2.resize(img2, (height * 5, width * 5))
    diff = cv2.resize(diff, (height * 5, width * 5))

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('diff', diff)
    cv2.waitKey(0)

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("请提供两个图像的文件路径")
    else:
        image1_path = sys.argv[1]
        image2_path = sys.argv[2]
        print(image1_path, image2_path)
        compare_images(image1_path, image2_path)
