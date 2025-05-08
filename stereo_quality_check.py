import cv2
import sys
import numpy as np

class StereoRectifier:
    def __init__(self):
        self._map_left = [None, None]  # 初始化为包含两个 None 的列表
        self._map_right = [None, None]  # 初始化为包含两个 None 的列表
        self.camera_left_matrix = None
        self.camera_T_stereo = None

    def init_from_calib(self, calib_config, image_size=None):
        fstereo = cv2.FileStorage(calib_config, cv2.FILE_STORAGE_READ)
        if not fstereo.isOpened():
            print(f"Open calib_config failed: {calib_config}")
            return False
        print(fstereo)
        node = fstereo.getNode("forward")
        if calib_config.endswith(".yaml"):
            width = int(node.getNode("cam0").getNode("width").real())
            height = int(node.getNode("cam0").getNode("height").real())
            print(f"width:{width}, height:{height}")
            camera_left = node.getNode("cam0").getNode("intrinsics").mat()
            dist_left = node.getNode("cam0").getNode("distortion_coeffs").mat()
            camera_right = node.getNode("cam1").getNode("intrinsics").mat()
            dist_right = node.getNode("cam1").getNode("distortion_coeffs").mat()
            mat_T_CL_CR = node.getNode("cam1").getNode("T_cn_cnm1").mat()
            R_stereo = mat_T_CL_CR[:3, :3]
            T_stereo = mat_T_CL_CR[:3, 3]

            self.camera_left_matrix = camera_left
            self.camera_T_stereo = T_stereo
            newImageSize = image_size if image_size else (width, height)

            R1, R2, P1, P2, Q, _, _  = cv2.stereoRectify(
                camera_left, dist_left, camera_right, dist_right, (width, height), R_stereo, T_stereo,
                cv2.CALIB_ZERO_DISPARITY, 0, newImageSize
            )

            self._map_left[0], self._map_left[1] = cv2.initUndistortRectifyMap(
                camera_left, dist_left, R1, P1, newImageSize, cv2.CV_32FC1
            )
            self._map_right[0], self._map_right[1] = cv2.initUndistortRectifyMap(
                camera_right, dist_right, R2, P2, newImageSize, cv2.CV_32FC1
            )
            return True
        else:
            print(f"Not supported camera calibrate config file: {calib_config}")
        return False

    def rectify(self, left, right):
        left_rectified = cv2.remap(left, self._map_left[0], self._map_left[1], cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, self._map_right[0], self._map_right[1], cv2.INTER_LINEAR)
        return left_rectified, right_rectified

def match_features(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2):
    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 匹配描述符
    matches = bf.match(descriptors1, descriptors2)
    filtered_matches = [
        match for match in matches
        if keypoints1[match.queryIdx].pt[0] > 0  # keypoints1[match.queryIdx].pt[0] 是关键点的 x 坐标
    ]
    # 根据距离排序
    filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)

    return filtered_matches[:50]  # 返回最好的50个匹配

def draw_matches(img1, img2, keypoints1, keypoints2, matches, text):
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 在图像上写文本
    cv2.putText(matched_img, text, (10, 30) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Matched Features', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if len(sys.argv) < 3:
    print("Usage: python script.py <left_img> <right_img> [calibration_file]")
    exit(1)

img1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

if len(sys.argv) == 4:
    rectifier = StereoRectifier()
    if rectifier.init_from_calib(sys.argv[3]):
        img1, img2 = rectifier.rectify(img1, img2)

orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# 特征匹配
matches = match_features(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2)

keypoints1_match = []
keypoints2_match = []
for i, match in enumerate(matches):
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    keypoints1_match.append(keypoints1[img1_idx])
    keypoints2_match.append(keypoints2[img2_idx])

# 计算y轴上的平均差值
total_y_diff = 0
for kp1, kp2 in zip(keypoints1_match, keypoints2_match):
    total_y_diff += abs(kp1.pt[1] - kp2.pt[1])
average_y_diff = total_y_diff / len(keypoints1_match)

# 可视化匹配结果
draw_matches(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), keypoints1, keypoints2, matches, f"y-axis diff: {average_y_diff}")
