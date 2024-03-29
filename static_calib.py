import cv2
import numpy as np
import sys
import math
from math import atan2, degrees
import os

debug = 0
def undistort_image(image, camera_matrix, distortion_coeffs):


    # 矫正畸变
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs)

    # 显示原始图像和矫正后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(1)
    return undistorted_image

def yaw_correction(yaw):
    if yaw > 90:
        yaw -= 180
    elif yaw < -90:
        yaw += 180
    return yaw

def detect_and_draw_corners(image, pattern_size, hground, real_dist):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # 绘制角点
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)

        # 显示带有角点的图像
        cv2.imshow("Chessboard Corners", image)

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
                if debug:
                    print("({},{})：({:.2f}, {:.2f})".format(x, y, cur_corner[0], cur_corner[1]))
                if x is not 0:
                    prev_corner = corner_coords[x-1,y,]
                    distance = np.linalg.norm(cur_corner - prev_corner)
                    center = (cur_corner + prev_corner) / 2
                    diff1[x-1,y] = [distance, center[0], center[1]]
                    if debug:
                        print("({:d},{:d}) ({:d},{:d}) distance：{:.2f}，center：({:.2f}, {:.2f})".format(x-1,y,x,y,distance, center[0], center[1]))
                if y is not 0:
                    prev_corner = corner_coords[x,y-1]
                    distance = np.linalg.norm(cur_corner - prev_corner)
                    center = (cur_corner + prev_corner) / 2
                    diff2[x,y-1] = [distance, center[0], center[1]]
                    if debug:
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
        focus = (distance_temp/distance_count) * hground / real_dist

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
        roll = degrees(math.atan(abs(diff_l_r_center)/diff_l_r_distance * hground / real_dist))
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
        pitch = degrees(math.atan(abs(diff_u_d_distance)/diff_u_d_center * hground / real_dist))
        print("focus:{} roll:{} pitch:{}".format(focus,roll,pitch))
    else:
        print("Chessboard corners not found.")

def detect_and_draw_circle_corners(image, pattern_size, hground, real_dist):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findCirclesGrid(gray, pattern_size, None)

    if ret:
        # 绘制角点
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)

        # 显示带有角点的图像
        cv2.imshow("Chessboard Corners", image)

        # 存储角点坐标和行列号
        corner_coords = np.zeros((pattern_size[0], pattern_size[1],2))
        diff1 = np.zeros((pattern_size[0]-1, pattern_size[1],3))
        diff2 = np.zeros((pattern_size[0], pattern_size[1]-1,3))

        for i, corner in enumerate(corners):
            x, y = corner[0]
            row = i // pattern_size[0]
            col = i % pattern_size[0]
            corner_coords[col, row] = [x, y]
        for x in range(pattern_size[0]):
            for y in range(pattern_size[1]):
                cur_corner = corner_coords[x, y]
                if debug:
                    print("({},{})：({:.2f}, {:.2f})".format(x, y, cur_corner[0], cur_corner[1]))
                if x is not 0:
                    prev_corner = corner_coords[x-1,y,]
                    distance = np.linalg.norm(cur_corner - prev_corner)
                    center = (cur_corner + prev_corner) / 2
                    diff1[x-1,y] = [distance, center[0], center[1]]
                    if debug:
                        print("({:d},{:d}) ({:d},{:d}) distance：{:.2f}，center：({:.2f}, {:.2f})".format(x-1,y,x,y,distance, center[0], center[1]))
                if y is not 0:
                    prev_corner = corner_coords[x,y-1]
                    distance = np.linalg.norm(cur_corner - prev_corner)
                    center = (cur_corner + prev_corner) / 2
                    diff2[x,y-1] = [distance, center[0], center[1]]
                    if debug:
                        print("({:d},{:d}) ({:d},{:d}) distance：{:.2f}，center：({:.2f}, {:.2f})".format(x,y-1,x,y,distance, center[0], center[1]))

        #计算焦距
        distance_temp = 0
        distance_count = 0
        for x in range(2,5):
            distance_temp = distance_temp + diff1[x,2][0]
            distance_count = distance_count + 1
        for y in range(1,4):
            distance_temp = distance_temp + diff2[2,y][0]
            distance_count = distance_count + 1
        focus = (distance_temp/distance_count) * hground / real_dist

        #计算yaw角
        x1 = corner_coords[0,0,0]
        y1 = corner_coords[0,0,1]
        x2 = corner_coords[0,pattern_size[1]-1,0]
        y2 = corner_coords[0,pattern_size[1]-1,1]
        x3 = corner_coords[pattern_size[0]-1,0,0]
        y3 = corner_coords[pattern_size[0]-1,0,1]
        x4 = corner_coords[pattern_size[0]-1,pattern_size[1]-1,0]
        y4 = corner_coords[pattern_size[0]-1,pattern_size[1]-1,1]

        yaw1 = degrees(atan2((x1+x2)/2 - (x3+x4)/2, (y1+y2)/2 - (y3+y4)/2))
        yaw2 = degrees(atan2((x1+x3)/2 - (x2+x4)/2, (y1+y3)/2 - (y2+y4)/2))
        if debug:
            print(yaw_correction(yaw1 - 90), yaw_correction(yaw2))
        yaw = -yaw_correction((yaw1 + yaw2 - 90) / 2)
        # 将角度限制在-90度到90度之间

        #
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
        #print(diff_l_r_distance, diff_l_r_center)
        roll = degrees(math.atan(abs(diff_l_r_distance)/diff_l_r_center * hground / real_dist))
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
        #print(diff_u_d_distance, diff_u_d_center)
        pitch = degrees(math.atan(abs(diff_u_d_distance)/diff_u_d_center * hground / real_dist))

        roll_real = roll * math.cos(yaw*math.pi/180) + pitch * math.sin(yaw*math.pi/180)
        pitch_real = -roll * math.sin(yaw*math.pi/180) + pitch * math.cos(yaw*math.pi/180)
        print("focus:{} yaw:{} roll:{} pitch:{} roll_real:{} pitch_real:{}".format(focus,yaw,roll,pitch,roll_real,pitch_real))
    else:
        print("Circleboard corners not found.")



def calib_single_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 输入相机的内参和畸变系数
    camera_matrix = np.array([[6.9164614728622178e+02, 0, 3.3349413481349200e+02],
                              [0, 6.9154334305690600e+02, 2.4024025772431770e+02],
                              [0, 0, 1]])
    distortion_coeffs = np.array([1.5906254649702670e-01, -1.7428984241764006e-01, 1.2726818077071658e-04, -2.0426778067850035e-03, 0])
    #distortion_coeffs = np.array([1.7110214481898720e-01, -2.2362607502949181e-01, 2.2898516428730212e-03, -2.0426778067850035e-03, 0])

    # 执行畸变矫正
    undistorted_image = undistort_image(image, camera_matrix, distortion_coeffs)

    # 设置棋盘格的大小
    pattern_size = (9, 6)  # 9行6列的棋盘格
    circle_pattern_size = (7, 5)  # 9行6列的棋盘格
    # 检测并绘制角点
    #detect_and_draw_corners(image, pattern_size, 250 ,24)
    detect_and_draw_circle_corners(undistorted_image, circle_pattern_size, 250, 30)
    cv2.waitKey(10)

image_extensions = ['.jpg', '.jpeg', '.png']  # 可以根据需要添加其他图像文件扩展名
def is_image_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension.lower() in image_extensions:
        return True
    else:
        return False

def traverse_images(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    sorted_image_files = sorted(image_files)  # 按照文件名进行排序

    for image_file in sorted_image_files:
        print("Image file:", image_file)
        # 在这里可以执行你想要的操作，比如读取图像、处理图像等
        calib_single_image(image_file)

input_path = sys.argv[1]
if len(sys.argv) > 2:
    output_path = sys.argv[2]
if is_image_file(input_path):
    calib_single_image(input_path)
else:
    traverse_images(input_path)

cv2.waitKey(0)
cv2.destroyAllWindows()
