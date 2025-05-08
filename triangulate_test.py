import cv2
import numpy as np

def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

# def compute_relative_rt(quaternion1, position1, quaternion2, position2, Tic):
#     R1 = quaternion_to_rotation_matrix(quaternion1)
#     R2 = quaternion_to_rotation_matrix(quaternion2)
#     p1 = np.array(position1)
#     p2 = np.array(position2)
#
#     # 计算相对旋转矩阵
#     R = R2.T.dot(R1)
#
#     # 计算相对平移向量
#     t = -R.dot(p2 - p1)
#
#     return R, t.reshape(-1, 1)

def compute_relative_rt(quaternion1, position1, quaternion2, position2, Tic):
    R1 = quaternion_to_rotation_matrix(quaternion1)
    R2 = quaternion_to_rotation_matrix(quaternion2)
    p1 = np.array(position1)
    p2 = np.array(position2)
    Tb0 = np.eye(4)
    Tb0[:3, :3] = R1
    Tb0[:3, 3] = p1
    Tb1 = np.eye(4)
    Tb1[:3, :3] = R2
    Tb1[:3, 3] = p2
    #T01 = np.linalg.inv((np.linalg.inv(Tic).dot(Tb0))).dot(np.linalg.inv(Tic).dot(Tb1))
    T01 = np.linalg.inv(Tb0.dot(Tic)).dot(Tb1.dot(Tic))
    R = T01[:3, :3]
    t = T01[:3, 3]
    return R, t.reshape(-1, 1)
#-5.124981880187988 -5.645709037780762 -9.959421157836914 0.029307996854186058 -0.005173438228666782 0.9915450811386108 0.126303568482399
#-11.516817092895508 -4.019191741943359 -9.957073211669922 0.006687205750495195 0.00041491937008686364 0.9919044971466064 0.12680891156196594
# 假设的四元数和位置向量
# quaternion1 = [0.029307996854186058, -0.005173438228666782, 0.9915450811386108, 0.126303568482399]  # 例如，单位四元数
# position1 = [-5.124981880187988 -5.645709037780762 -9.959421157836914]  # 例如，原点
# quaternion2 = [0.037661779671907425, -0.0030404909048229456, 0.9909967184066772, 0.12844403088092804]  # 例如，沿x轴旋转90度
# position2 = [-6.297181606292725, -5.325246334075928, -9.960309028625488]  # 例如，沿x轴移动1单位
quaternion1 = [-0.005522096995264292, 0.05953050032258034, 0.49538347125053406, -0.8666145205497742]  # 例如，单位四元数
position1 = [-128.1279296875, 240.2775421142578, -147.17013549804688]  # 例如，原点
quaternion2 = [-0.005182153545320034, 0.06742226332426071, 0.4953703284263611, -0.8660459518432617]  # 例如，沿x轴旋转90度
position2 = [-127.00553131103516, 238.3629913330078, -147.17141723632812]  # 例如，沿x轴移动1单位


Tic = np.array([
    [9.8869110000000000e-03, 9.9994689999999997e-01, -2.9113460000000000e-03, 1.7103170000000001e-01],
    [-9.9995109999999998e-01, 9.8860860000000005e-03, -2.9781580000000000e-04, 6.4451480000000005e-02],
    [-2.6901820000000001e-04, 2.9141480000000001e-03, 9.9999570000000004e-01, 8.6214990000000005e-02],
    [0., 0., 0.,1.]
])

# 相机内参矩阵K
#forward
# K = np.array([
#     [677.6927, 0., 636.97369],
#     [0., 678.05182, 343.62585],
#     [0., 0., 1.]
# ])
#downward
K = np.array([
    [678.1033, 0., 663.380],
    [0., 678.5874, 355.262],
    [0., 0., 1.]
])
D = np.array([ -5.5829669631635180e-02, -7.0436236296433476e-03, -2.0686728584031803e-04, -4.0326570630819643e-04, 0. ])  # 畸变系数

R, t = compute_relative_rt(quaternion1, position1, quaternion2, position2, Tic)

print("Relative Rotation Matrix:")
print(R)
print("Relative Translation Vector:")
print(t)

def match_features(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2):
    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 匹配描述符
    matches = bf.match(descriptors1, descriptors2)
    # 根据距离排序
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:200]  # 返回最好的50个匹配
def draw_matches(img1, img2, keypoints1, keypoints2, matches):
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matched Features', matched_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 读取图像并提取特征
# img1 = cv2.imread("1724319648811596968.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("1724319650411676072.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("1724327351917715664.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("1724327353086132720.png", cv2.IMREAD_GRAYSCALE)
# undistort image
img1 = cv2.undistort(img1, K, D)
img2 = cv2.undistort(img2, K, D)

orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# 特征匹配
matches = match_features(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2)
# 可视化匹配结果
draw_matches(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), keypoints1, keypoints2, matches)
print(len(keypoints1))
def pixel2cam(point, K):
    x, y = point
    point_norm = np.array([x, y])
    return point_norm
    # #return np.array([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1]])
    # print(f"x: {x}")
    # print(f"y: {y}")
    # return np.array(x, y)

def triangulatePoints(K, R, t, pts1, pts2):
    # 构建投影矩阵
    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float32)
    T2 = np.hstack((R, t))
    P1 = np.matmul(K, T1)
    P2 = np.matmul(K, T2)

    # 三角化
    print(P1)
    print(P2)
    print(pts1.T.shape)
    print(pts2.T.shape)
    pts_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_4d = pts_4d_hom / pts_4d_hom[3, :]  # 正确地除以最后一列

    return pts_4d.T

def draw_triangulated_points(img1, img2, keypoints1, keypoints2, triangulated_pts):
    # 绘制匹配线
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    for i, points in enumerate(keypoints1):
        print(f"i: {i}")
        x1, y1 = keypoints1[i].pt
        x2, y2 = keypoints2[i].pt
        # x2 += w1
        # y2 += h1
        color = (0, 255, 0)
        thickness = 1
        cv2.line(vis, (int(x1), int(y1)), (int(x2 + w1), int(y2)), color, thickness)
        cv2.circle(vis, (int(x1), int(y1)), 2, (255, 0, 0), -1)
        cv2.circle(vis, (int(x2 + w1), int(y2)), 2, (255, 0, 0), -1)

        # 绘制三角化点
        pt_3d = triangulated_pts[i]
        # print(f"pt_3d shape: {pt_3d.shape}")
        x3d, y3d, z3d, _ = pt_3d
        cv2.circle(vis, (int(x1), int(y1)), 5, (0, 0, 255), -1)
        #cv2.putText(vis, f"({x3d:.2f}, {y3d:.2f}, {z3d:.2f})", (int(x1) + 10, int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis, f"({x3d:.2f}, {y3d:.2f}, {z3d:.2f})", (int(x1) + 10, int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"pt_3d: {pt_3d}")
    cv2.imshow('Triangulated Points', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def draw_triangulated_points_3d(triangulated_pts):
    # 检查点云数量
    if triangulated_pts.size == 0:
        print("No triangulated points to display.")
        return
    filtered_pts = triangulated_pts[(np.abs(triangulated_pts[:, 0]) < 1000) &
                                    (np.abs(triangulated_pts[:, 1]) < 1000) &
                                    (np.abs(triangulated_pts[:, 2]) < 1000)]
    # 提取3D点的坐标
    x = filtered_pts[:, 0]
    y = filtered_pts[:, 1]
    z = filtered_pts[:, 2]
    print()

    # 创建一个新的图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D点云
    ax.scatter(x, y, z, c='r', marker='o')

    # 设置图表标题和坐标轴标签
    ax.set_title('Triangulated 3D Points')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # 显示图表
    plt.show()


# 将特征点坐标转换为归一化坐标
keypoints1_match = []
keypoints2_match = []
for i, match in enumerate(matches):
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    keypoints1_match.append(keypoints1[img1_idx])
    keypoints2_match.append(keypoints2[img2_idx])


pts1_cam = np.array([pixel2cam(kp.pt, K) for kp in keypoints1_match])
pts2_cam = np.array([pixel2cam(kp.pt, K) for kp in keypoints2_match])

# # 转置特征点坐标
# pts1_cam = pts1_cam.T
# pts2_cam = pts2_cam.T

# 三角化
triangulated_pts = triangulatePoints(K, R, t, pts1_cam, pts2_cam)

# 可视化三角化点
print(len(triangulated_pts))
draw_triangulated_points(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), keypoints1_match, keypoints2_match, triangulated_pts)
print(len(triangulated_pts))


# R_0 = quaternion_to_rotation_matrix(quaternion1)
# t_0 = np.array(position1)
# T = np.eye(4)
# T[:3, :3] = R_0
# world_pts = Tic.dot(triangulated_pts.T)
draw_triangulated_points_3d(triangulated_pts)