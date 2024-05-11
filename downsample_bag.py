import argparse
import rosbag
import cv2

def downsample_bag_topics(input_bag, output_bag, left_topic, right_topic, factor):
    # 打开输入和输出bag文件
    left_timestamp = None
    left_msg = None
    right_timestamp = None
    right_msg = None
    count = 0
    with rosbag.Bag(input_bag, 'r') as inbag, rosbag.Bag(output_bag, 'w') as outbag:
        # 遍历输入bag中的每个消息
        for topic, msg, t in inbag.read_messages():
            if topic == left_topic:
                count += 1
                left_timestamp = msg.header.stamp
                left_msg = msg
            elif topic == right_topic:
                right_timestamp = msg.header.stamp
                right_msg = msg
            else:
                outbag.write(topic, msg, msg.header.stamp)
                print("write {}! time:{}".format(topic,msg.header.stamp))
                continue
            #print(left_timestamp, right_timestamp)
            if left_timestamp is not None and right_timestamp is not None and left_timestamp == right_timestamp and left_msg is not None and right_msg is not None:
                if count % factor == 0:
                    outbag.write(left_topic, left_msg, msg.header.stamp)
                    outbag.write(right_topic, right_msg, msg.header.stamp)
                    left_timestamp = None
                    left_msg = None
                    right_timestamp = None
                    right_msg = None
                    print("write stereo! time:",msg.header.stamp)


    print("抽帧完成！")

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='对ROS bag文件中的所有topic进行抽帧')
    parser.add_argument('input_bag', type=str, help='输入的ROS bag文件路径')
    parser.add_argument('output_bag', type=str, help='输出的ROS bag文件路径')
    parser.add_argument('left_topic', type=str, help='输入的ROS bag文件路径')
    parser.add_argument('right_topic', type=str, help='输入的ROS bag文件路径')
    parser.add_argument('factor', type=float, help='降低的倍数')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数进行抽帧
    downsample_bag_topics(args.input_bag, args.output_bag, args.left_topic, args.right_topic, args.factor)
