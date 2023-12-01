import os
import sys
import cv2
import rosbag
from cv_bridge import CvBridge

def extract_images_from_bag(bag_path, interval, left_topic):
    # 创建保存图像的文件夹
    base_folder = os.path.splitext(os.path.basename(bag_path))[0]
    output_folder = os.path.join(os.path.dirname(bag_path), base_folder)
    os.makedirs(output_folder, exist_ok=True)
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
    for topic, msg, t in bag.read_messages(topics=left_topic):
        if topic == left_topic:
            count += 1
            if count % interval == 0:
                left_timestamp = msg.header.stamp
                print(left_timestamp)
                left_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                left_img_path = os.path.join(output_folder, f"{left_timestamp}.png")
                cv2.imwrite(left_img_path, left_image)

    # 关闭 ROS bag 文件
    bag.close()

if __name__ == '__main__':
    # 获取输入参数
    # usage: python parse_single_img.py /mnt/1t_1/data/zed/2023-08-31-17-21-26.bag 10 /zed/zed_node/left/image_rect_color
    bag_path = sys.argv[1]  # bag 文件路径
    interval = int(sys.argv[2])  # 抽取帧数的间隔
    left_topic = sys.argv[3]  # 左图像的话题

    # 调用函数提取图像
    extract_images_from_bag(bag_path, interval, left_topic)