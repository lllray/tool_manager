import os
import sys
import cv2
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import numpy as np

def extract_images_from_bag(bag_path, interval, left_topic, right_topic):
    # 创建保存图像的文件夹
    base_folder = os.path.splitext(os.path.basename(bag_path))[0]
    output_folder = os.path.join(os.path.dirname(bag_path), base_folder)
    left_folder = os.path.join(output_folder, 'left')
    right_folder = os.path.join(output_folder, 'right')
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)

    # 打开 ROS bag 文件
    bag = rosbag.Bag(bag_path)

    # 初始化图像计数器
    count = 0

    # 初始化 CvBridge
    bridge = CvBridge()

    # 初始化时间戳和图像变量
    left_timestamp = None
    left_image = None
    right_timestamp = None
    right_image = None

    # 遍历 bag 文件中的消息
    for topic, msg, t in bag.read_messages(topics=[left_topic, right_topic]):
        if topic == left_topic:
            count += 1
            left_timestamp = msg.header.stamp
            #print("left encoding:",msg.encoding)
            if 'format' in dir(msg):
                np_arr = np.fromstring(msg.data, np.uint8)
                left_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif msg.encoding == 'bgr8':
                #print("left encoding:",msg.encoding)
                left_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        elif topic == right_topic:
            right_timestamp = msg.header.stamp
            #print("right encoding:",msg.encoding)
            if 'format' in dir(msg):
                np_arr = np.fromstring(msg.data, np.uint8)
                right_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif msg.encoding == 'bgr8':
                #print("right encoding:",msg.encoding)
                right_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        print(left_timestamp, right_timestamp)
        # 检查左右图像时间戳是否一致且图像不为空
        if left_timestamp is not None and right_timestamp is not None and left_timestamp == right_timestamp and left_image is not None and right_image is not None:
            if count % interval == 0 and int(left_timestamp.secs) != 0:
                # 保存左图像
                left_img_path = os.path.join(left_folder, f"{left_timestamp}.png")
                print("left image path:",left_img_path)
                cv2.imwrite(left_img_path, left_image)

                # 保存右图像
                right_img_path = os.path.join(right_folder, f"{right_timestamp}.png")
                print("right image path:",right_img_path)
                cv2.imwrite(right_img_path, right_image)
                # 重置时间戳和图像变量
                left_timestamp = None
                left_image = None
                right_timestamp = None
                right_image = None




    # 关闭 ROS bag 文件
    bag.close()

if __name__ == '__main__':
    # 获取输入参数
    # usage: python parse_stereo_img.py /mnt/1t_1/data/zed/2023-08-31-17-21-26.bag 10 /zed/zed_node/left/image_rect_color /zed/zed_node/right/image_rect_colo
    bag_path = sys.argv[1]  # bag 文件路径
    interval = int(sys.argv[2])  # 抽取帧数的间隔
    left_topic = sys.argv[3]  # 左图像的话题
    right_topic = sys.argv[4]  # 右图像的话题

    # 调用函数提取图像
    extract_images_from_bag(bag_path, interval, left_topic, right_topic)
