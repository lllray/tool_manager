import cv2
import numpy as np

# 读取两个输入视频文件
video1 = cv2.VideoCapture('../euroc_mh_01.mp4')
video2 = cv2.VideoCapture('vis.stab_euroc_mh_01_img.mp4')

# 获取第一个视频的宽度和高度
width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建输出视频的写入器
output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width*2, height))

while True:
    # 读取第一个视频的帧
    ret1, frame1 = video1.read()
    if not ret1:
        break
    
    # 读取第二个视频的帧
    ret2, frame2 = video2.read()
    if not ret2:
        break
    
    # 将两个帧横向合并
    merged_frame = np.hstack((frame1, frame2))
    
    # 写入合并后的帧到输出视频文件
    output.write(merged_frame)

# 释放资源
video1.release()
video2.release()
output.release()
