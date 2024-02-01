import cv2
import numpy as np
import sys
import math
def undistort_image(image, camera_matrix, distortion_coeffs):


    # 矫正畸变
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs)

    # 显示原始图像和矫正后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(1)
    return undistorted_image

def detect_and_draw_corners(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # 绘制角点
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)

        # 显示带有角点的图像
        cv2.imshow("Chessboard Corners", image)
        cv2.waitKey(1)

        # 存储角点坐标和行列号
        corner_coords = np.zeros((pattern_size[0], pattern_size[1],2))
        diff1 = np.zeros((pattern_size[0]-1, pattern_size[1],3))
        diff2 = np.zeros((pattern_size[0], pattern_size[1]-1,3))

        for i, corner in enumerate(corners):
            x, y = corner[0]
            row = i // pattern_size[0]
            col = i % pattern_size[0]
            corner_coords[pattern_size[0] - col - 1, pattern_size[1] - row - 1] = [x, y]
        for x in range(pattern_size[0]):
            for y in range(pattern_size[1]):
                cur_corner = corner_coords[x, y]
                print("({},{})：({:.2f}, {:.2f})".format(x, y, cur_corner[0], cur_corner[1]))
                if x is not 0:
                    prev_corner = corner_coords[x-1,y,]
                    distance = np.linalg.norm(cur_corner - prev_corner)
                    center = (cur_corner + prev_corner) / 2
                    diff1[x-1,y] = [distance, center[0], center[1]]
                    print("({:d},{:d}) ({:d},{:d}) distance：{:.2f}，center：({:.2f}, {:.2f})".format(x-1,y,x,y,distance, center[0], center[1]))
                if y is not 0:
                    prev_corner = corner_coords[x,y-1]
                    distance = np.linalg.norm(cur_corner - prev_corner)
                    center = (cur_corner + prev_corner) / 2
                    diff2[x,y-1] = [distance, center[0], center[1]]
                    print("({:d},{:d}) ({:d},{:d}) distance：{:.2f}，center：({:.2f}, {:.2f})".format(x,y-1,x,y,distance, center[0], center[1]))

        #计算焦距
        distance_temp = 0
        distance_count = 0
        for x in range(3,5):
            for y in range(2,4):
                distance_temp = distance_temp + diff1[x,y][0]
                distance_count = distance_count + 1
        for x in range(3,6):
            distance_temp = distance_temp + diff2[x,2][0]
            distance_count = distance_count + 1

        print("focus: {}".format((distance_temp/distance_count) * 250 / 24))

        offset_max = 1
        # 计算 roll
        left_distance_temp = 0
        left_center = 0
        right_distance_temp = 0
        right_center = 0
        count = 0
        for y in range(pattern_size[1]-1):
            for offset in range(offset_max):
                left_distance_temp = left_distance_temp + diff2[0 + offset,y][0]
                left_center = left_center + diff2[0 + offset,y][1]
                right_distance_temp = right_distance_temp + diff2[pattern_size[0]-1 - offset,y][0]
                right_center = right_center + diff2[pattern_size[0]-1 - offset,y][1]
                count = count + 1
                #print(diff1[x,0][0], diff2[pattern_size[0]-1,y][0])
                #print(diff2[0,y][1], diff2[pattern_size[0]-1,y][1])
        diff_l_r_distance = (right_distance_temp - left_distance_temp)/count
        diff_l_r_center = (right_center - left_center)/count
        print(diff_l_r_distance, diff_l_r_center)
        print("roll:",math.atan(abs(diff_l_r_distance)/diff_l_r_center * 250 / 24) * 180 / 3.1415926)
        # 计算pitch
        up_distance_temp = 0
        up_center = 0
        down_distance_temp = 0
        down_center = 0
        count = 0
        for x in range(pattern_size[0]-1):
            for offset in range(offset_max):
                up_distance_temp = up_distance_temp + diff1[x,0 + offset][0]
                up_center = up_center + diff1[x,0 + offset][2]
                down_distance_temp = down_distance_temp + diff1[x, pattern_size[1]-1 - offset][0]
                down_center = down_center + diff1[x, pattern_size[1]-1 - offset][2]
                count = count + 1
                #print(diff1[x,0][0], diff1[x, pattern_size[1]-1][0])
                #print(diff1[x,0][2], diff1[x, pattern_size[1]-1][2])
        diff_u_d_distance = (down_distance_temp - up_distance_temp)/count
        diff_u_d_center = (down_center -up_center)/count
        print(diff_u_d_distance, diff_u_d_center)
        print("pitch:",math.atan(abs(diff_u_d_distance)/diff_u_d_center * 250 / 24 ) * 180 / 3.1415926)

    else:
        print("Chessboard corners not found.")

# 输入图像路径
image_path = sys.argv[1]
# 读取图像
image = cv2.imread(image_path)

# 创建锐化滤波器
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# 应用滤波器
sharpened_image = cv2.filter2D(image, -1, kernel)
# 输入相机的内参和畸变系数
camera_matrix = np.array([[6.9164614728622178e+02, 0, 3.3349413481349200e+02],
                          [0, 6.9154334305690600e+02, 2.4024025772431770e+02],
                          [0, 0, 1]])
distortion_coeffs = np.array([1.5906254649702670e-01, -1.7428984241764006e-01, 1.2726818077071658e-04, -2.0426778067850035e-03, 0])

# 执行畸变矫正
undistorted_image = undistort_image(sharpened_image, camera_matrix, distortion_coeffs)

# 设置棋盘格的大小
pattern_size = (9, 6)  # 9行6列的棋盘格
# 检测并绘制角点
detect_and_draw_corners(sharpened_image, pattern_size)
cv2.waitKey(0)
cv2.destroyAllWindows()