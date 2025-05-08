import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d

def create_fov_cone(position, direction, fov_angle=60, max_range=5, color='r', alpha=0.2):
    """创建视场角锥体"""
    # 生成锥体网格
    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(0, max_range, 30)
    theta, z = np.meshgrid(theta, z)
    r = z * np.tan(np.deg2rad(fov_angle / 2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 根据方向向量调整网格
    axis = np.array(direction)
    axis = axis / np.linalg.norm(axis)  # 单位化方向向量

    # 特殊处理负 Z 轴方向
    if np.allclose(axis, [0, 0, -1]):
        z = -z  # 翻转 Z 方向
    else:
        # 将网格点投影到指定方向
        points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        if not np.allclose(axis, [0, 0, 1]):  # 如果方向不是默认的 Z 轴
            rotation_axis = np.cross([0, 0, 1], axis)
            rotation_angle = np.arccos(np.dot([0, 0, 1], axis))
            rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)
            points = np.dot(rotation_matrix, points)
        x, y, z = points[0].reshape(x.shape), points[1].reshape(y.shape), points[2].reshape(z.shape)

    # 绘制锥体
    ax.plot_surface(x + position[0], y + position[1], z + position[2],
                    color=color, alpha=alpha)

def create_fov_hemisphere(position, direction, fov_angle=180, max_range=5, color='r', alpha=0.2):
    """创建半圆视场角"""
    # 生成半圆网格
    theta = np.linspace(0, np.pi, 30)  # 半圆角度范围 [0, π]
    phi = np.linspace(0, np.pi, 30)   # 垂直角度范围
    theta, phi = np.meshgrid(theta, phi)
    r = max_range
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # 旋转对齐方向
    axis = np.array(direction)
    axis = axis / np.linalg.norm(axis)  # 单位化方向向量
    if not np.allclose(axis, [0, 0, 1]):  # 如果方向不是默认的 Z 轴
        rotation_axis = np.cross([0, 0, 1], axis)
        rotation_angle = np.arccos(np.dot([0, 0, 1], axis))
        rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)
        points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        rotated_points = np.dot(rotation_matrix, points)
        x, y, z = rotated_points[0].reshape(x.shape), rotated_points[1].reshape(y.shape), rotated_points[2].reshape(z.shape)

    # 绘制半圆
    ax.plot_surface(x + position[0], y + position[1], z + position[2],
                    color=color, alpha=alpha)

def create_rectangular_fov_cone(position, direction, up_vector, hfov_angle=60, vfov_angle=40, max_range=5, color='r', alpha=0.2):
    """
    创建底面为矩形的视场角锥体，支持自由度更高的方向控制。
    
    参数:
        position (list or np.ndarray): 相机的位置 [x, y, z]。
        direction (list or np.ndarray): 相机的指向方向向量 [dx, dy, dz]。
        up_vector (list or np.ndarray): 相机的上方向向量 [ux, uy, uz]，用于定义转轴方向。
        hfov_angle (float): 水平视场角（角度）。
        vfov_angle (float): 垂直视场角（角度）。
        max_range (float): 视场的最大范围。
        color (str): 锥体的颜色。
        alpha (float): 锥体的透明度。
    """
    # 确保输入是浮点数类型
    position = np.array(position, dtype=np.float64)
    direction = np.array(direction, dtype=np.float64)
    up_vector = np.array(up_vector, dtype=np.float64)

    # 单位化方向向量和上方向向量
    direction = direction / np.linalg.norm(direction)
    up_vector = up_vector / np.linalg.norm(up_vector)

    # 计算矩形底面的边界点
    half_width = max_range * np.tan(np.deg2rad(hfov_angle / 2))
    half_height = max_range * np.tan(np.deg2rad(vfov_angle / 2))

    # 定义矩形底面的四个顶点（在局部坐标系中）
    base_points = np.array([
        [-half_width, -half_height, max_range],  # 左下角
        [half_width, -half_height, max_range],   # 右下角
        [half_width, half_height, max_range],    # 右上角
        [-half_width, half_height, max_range]    # 左上角
    ])

    # 构造相机的局部坐标系
    forward = direction  # 指向方向
    right = np.cross(forward, up_vector)  # 右方向
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)  # 上方向（重新计算以确保正交）

    # 构造旋转矩阵，将局部坐标系对齐到全局坐标系
    rotation_matrix = np.array([right, up, forward]).T

    # 将矩形底面的点从局部坐标系转换到全局坐标系
    base_points = np.dot(base_points, rotation_matrix.T)

    # 添加锥体的顶点（即相机的位置）
    apex = np.array([0, 0, 0], dtype=np.float64)  # 锥体顶点在局部坐标系中为原点

    # 将所有点平移到全局坐标系
    base_points += position
    apex += position

    # 创建锥体的表面
    vertices = np.vstack([apex, base_points])  # 合并顶点和底面顶点
    faces = [
        [0, 1, 2],  # 顶点到左下角和右下角
        [0, 2, 3],  # 顶点到右下角和右上角
        [0, 3, 4],  # 顶点到右上角和左上角
        [0, 4, 1],  # 顶点到左上角和左下角
        [1, 2, 3, 4]  # 底面矩形
    ]

    # 绘制锥体
    poly3d = [[vertices[i] for i in face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=0.1, edgecolors=color, alpha=alpha))

def rotation_matrix_from_axis_angle(axis, angle):
    """从轴和角度生成旋转矩阵"""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

# 创建3D画布
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制无人机主体（立方体）
drone_size = [0.8, 0.8, 0.5]
drone_pos = [-drone_size[0]/2, -drone_size[1]/2, 0]  # 左下角位置
vertices = [
    [drone_pos[0], drone_pos[1], drone_pos[2]],
    [drone_pos[0] + drone_size[0], drone_pos[1], drone_pos[2]],
    [drone_pos[0] + drone_size[0], drone_pos[1] + drone_size[1], drone_pos[2]],
    [drone_pos[0], drone_pos[1] + drone_size[1], drone_pos[2]],
    [drone_pos[0], drone_pos[1], drone_pos[2] + drone_size[2]],
    [drone_pos[0] + drone_size[0], drone_pos[1], drone_pos[2] + drone_size[2]],
    [drone_pos[0] + drone_size[0], drone_pos[1] + drone_size[1], drone_pos[2] + drone_size[2]],
    [drone_pos[0], drone_pos[1] + drone_size[1], drone_pos[2] + drone_size[2]]
]
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
    [vertices[0], vertices[3], vertices[7], vertices[4]]   # 左面
]
cube = Poly3DCollection(faces, facecolors='gray', linewidths=1, edgecolors='black', alpha=0.2)
ax.add_collection3d(cube)

# 添加螺旋桨（圆形）
prop_positions = [[0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]]
for pos in prop_positions:
    circle = plt.Circle((pos[0], pos[1]), 0.3, color='black', fill=True)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=pos[2], zdir="z")

#添加激光雷达（顶部）
# lidar_pos = [0, 0, drone_size[2]]
# create_fov_hemisphere(lidar_pos, [0, -90, 1], fov_angle=180, color='green', max_range=5, alpha=0.2)

camera_positions = [[0, 0.5, 0.5], [0, 0.5, 0.5]]  # 前向和向下
camera_directions = [[-0.466, 1, 0], [0.466, 1, 0]]  # 前向和向下
camera_up_vectors = [[0, 0, 1], [0, 0, 1]]  # 不同的上方向

for pos, dir, up in zip(camera_positions, camera_directions, camera_up_vectors):
    create_rectangular_fov_cone(pos, dir, up, hfov_angle=130, vfov_angle=107, color='blue', max_range=3, alpha=0.05)
#
# # camera_positions = [[0, 0.5, 0.5], [0, 0.5, 0.5]]  # 前向和向下
# # camera_directions = [ [0, 0.839, -1], [0, 0.839, 1]]  # 前向和向下
# # camera_up_vectors = [ [0, 0, 1], [0, 1, 1]]  # 不同的上方向
camera_positions = [[0, 0.5, 0.5]]  # 前向和向下
camera_directions = [ [0, 0.839, -2]]  # 前向和向下
camera_up_vectors = [ [0, 0, 1]]  # 不同的上方向

for pos, dir, up in zip(camera_positions, camera_directions, camera_up_vectors):
    create_rectangular_fov_cone(pos, dir, up, hfov_angle=130, vfov_angle=107, color='blue', max_range=3, alpha=0.05)


# camera_positions = [[0, 0.5, 0.5], [0, 0.5, 0.5], [0, 0.5, 0.5], [0, 0.5, 0.5]]  # 前向和向下
# camera_directions = [[-0.466, 1, 1],  [-0.466, 1, -1], [0.466, 1, 1],  [0.466, 1, -1]]  # 前向和向下
# camera_up_vectors = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]  # 不同的上方向
#
# for pos, dir, up in zip(camera_positions, camera_directions, camera_up_vectors):
#     create_rectangular_fov_cone(pos, dir, up, hfov_angle=130, vfov_angle=107, color='blue', max_range=3, alpha=0.05)


# # 添加毫米波雷达（两侧）
# radar_positions = [[0.5, 0, drone_size[2]], [-0.5, 0, drone_size[2]]]
# for pos in radar_positions:
#     create_fov_hemisphere(pos, [0, 0, 1], fov_angle=180, color='red', max_range=4)

# create_fov_cone(position=[0, 0, 0], direction=[0, 0, 1], fov_angle=90, color='green')  # 向上
# create_fov_cone(position=[0, 0, 0], direction=[0, 0, -1], fov_angle=90, color='blue')  # 向下
# create_fov_cone(position=[0, 0, 0], direction=[1, 0, 0], fov_angle=90, color='red')  # 向右

# 设置坐标轴
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-3, 7)

plt.show()
