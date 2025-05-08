import os
import sys
import cv2
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import numpy as np
def extract_images_from_bag(bag_path, interval, left_topic, imu_topic):
    # 创建保存图像的文件夹
    base_folder = os.path.splitext(os.path.basename(bag_path))[0]
    output_folder = os.path.join(os.path.dirname(bag_path), base_folder)
    os.makedirs(output_folder, exist_ok=True)

    imu_output_txt = os.path.join(output_folder, "imu.txt")
    img_output_txt = os.path.join(output_folder, "img.txt")
    img_output_file = os.path.join(output_folder, "img")
    output_video = os.path.join(output_folder, "video.mp4")
    os.makedirs(img_output_file, exist_ok=True)

    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = None

    img_file = open(img_output_txt, 'w')
    if imu_topic != None:
        imu_file = open(imu_output_txt, 'w')

    # 打开 ROS bag 文件
    bag = rosbag.Bag(bag_path)

    # 初始化图像计数器
    count = 0


    start_count = 0

    # 初始化 CvBridge
    bridge = CvBridge()

    # 初始化时间戳和图像变量
    left_timestamp = None
    left_image = None
    right_timestamp = None
    right_image = None

    # 遍历 bag 文件中的消息
    for topic, msg, t in bag.read_messages():
        if topic == left_topic:
            count += 1
            print("count:",count)
            if count % interval == 0 and count >= start_count:
                left_timestamp = msg.header.stamp.to_nsec()
                print(left_timestamp)
                if 'format' in dir(msg):
                    np_arr = np.fromstring(msg.data, np.uint8)
                    left_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                elif msg.encoding == 'bgr8':
                    #print("left encoding:",msg.encoding)
                    left_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                if video is None:
                    height, width, _ = left_image.shape
                    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
                left_img_path = os.path.join(img_output_file, f"{left_timestamp}.png")
                cv2.imwrite(left_img_path, left_image)
                video.write(left_image)
                img_file.write(f"{left_timestamp} 100000\n")
        if imu_topic != None and topic == imu_topic and count >= start_count:
            timestamp = msg.header.stamp.to_nsec()
            gx = msg.angular_velocity.x
            gy = msg.angular_velocity.y
            gz = msg.angular_velocity.z
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z

            # 将数据写入文本文件中
            line = f"{timestamp} {gx} {gy} {gz} {ax} {ay} {az}\n"
            imu_file.write(line)

    # 关闭 ROS bag 文件
    imu_file.close()
    img_file.close()
    video.release()
    bag.close()

if __name__ == '__main__':
    # 获取输入参数
    # usage: python parse_single_img.py /mnt/1t_1/data/zed/2023-08-31-17-21-26.bag 10 /zed/zed_node/left/image_rect_color
    bag_path = sys.argv[1]  # bag 文件路径
    interval = int(sys.argv[2])  # 抽取帧数的间隔
    left_topic = sys.argv[3]  # 左图像的话题
    if len(sys.argv) > 4:
        imu_topic = sys.argv[4]
    else:
        imu_topic = None

    # 调用函数提取图像
    extract_images_from_bag(bag_path, interval, left_topic, imu_topic)
