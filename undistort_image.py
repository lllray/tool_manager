import cv2
import numpy as np

def undistort_image(image_path, camera_matrix, distortion_coeffs):
    # 读取图像
    image = cv2.imread(image_path)

    # 矫正畸变
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs)

    # 显示原始图像和矫正后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 输入图像路径
image_path = "image.jpeg"

# 输入相机的内参和畸变系数
camera_matrix = np.array([[6.9164614728622178e+02, 0, 3.3349413481349200e+02],
                          [0, 6.9154334305690600e+02, 2.4024025772431770e+02],
                          [0, 0, 1]])
distortion_coeffs = np.array([1.5906254649702670e-01, -1.7428984241764006e-01, 1.2726818077071658e-04, -2.0426778067850035e-03, 0])

# 执行畸变矫正
undistort_image(image_path, camera_matrix, distortion_coeffs)
