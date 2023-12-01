import numpy as np
import sys
import re
from transforms3d import quaternions, affines
# 读取pose数据
if len(sys.argv) < 2:
    print("Please provide the path to the log file as a command line argument.")
    sys.exit(1)

log_file_path = sys.argv[1]

with open(log_file_path, "r") as f:
    data = f.read()

blocks = re.split("---\n", data.strip())
poses = []
last_key = []
for block in blocks:
    #print(block)
    lines = block.strip().split("\n")
    item = {}

    for line in lines:
        key, value = re.findall(r"(\w+):\s*(.*)", line)[0]
        #print(key,value)
        if key == "secs" or key == "nsecs":
            item[f"{last_key}_{key}"] = float(value)
        elif key == "x" or key == "y" or key == "z" or key == "w":
            item[f"{last_key}_{key}"] = float(value)
        else :
            last_key = key
    #print(item)
    poses.append(item)

output = []
velocities = []
timestamps = []
for i, pose in enumerate(poses):
    timestamps.append(pose['stamp_secs'] * 1000 + pose['stamp_nsecs'] / 1e6)
    if i == 0:
        velocities.append((0, 0))
        continue

    prev_pose = poses[i - 1]
    curr_pose = poses[i]
    # 计算时间间隔
    time_diff = timestamps[i] - timestamps[i - 1]

    # 构建姿态1的变换矩阵
    transformation_matrix1 = affines.compose([prev_pose['position_x'], prev_pose['position_y'], prev_pose['position_z']], quaternions.quat2mat([prev_pose['orientation_w'], prev_pose['orientation_x'], prev_pose['orientation_y'], prev_pose['orientation_z']]), [1.0, 1.0, 1.0])

    # 构建姿态2的变换矩阵
    transformation_matrix2 = affines.compose([curr_pose['position_x'], curr_pose['position_y'], curr_pose['position_z']], quaternions.quat2mat([curr_pose['orientation_w'], curr_pose['orientation_x'], curr_pose['orientation_y'], curr_pose['orientation_z']]), [1.0, 1.0, 1.0])

    # 计算姿态2相对于姿态1的位移向量
    displacement_vector = transformation_matrix2.dot(np.linalg.inv(transformation_matrix1))
    # print(transformation_matrix2.dot(np.linalg.inv(transformation_matrix1)))
    # 计算速度
    vx = displacement_vector[0,3]
    vy = displacement_vector[1,3]
    #
    velocities.append((vx, vy))

# 输出结果到txt文本
output_file = open('../openmv_log/23-10-26/zed_pose_parse_v.txt', 'w')
for i in range(len(poses)):
    print("TIME: {0:.6f}, VX: {1:.6f}, VY: {2:.6f}\n".format(timestamps[i], velocities[i][0], velocities[i][1]))
    output_file.write("TIME: {0:.6f}, VX: {1:.6f}, VY: {2:.6f}\n".format(timestamps[i], velocities[i][0], velocities[i][1]))
output_file.close()