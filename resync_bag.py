import argparse
import rosbag
import cv2
import sys

from rospy import Duration
def resync_bag_topics(input_bag, output_bag):
    with rosbag.Bag(input_bag, 'r') as inbag, rosbag.Bag(output_bag, 'w') as outbag:
        # 遍历输入bag中的每个消息
        for topic, msg, t in inbag.read_messages():
            if topic == "/fvs/drone/imu_raw":
                msg.header.stamp = msg.header.stamp + Duration.from_sec(-0.017)
            outbag.write(topic, msg, msg.header.stamp)
            #print("write {}! time:{}".format(topic,msg.header.stamp))
            continue

    print("resync完成！")

if __name__ == '__main__':

    input_bag = sys.argv[1]
    suffix = ".bag"
    output_bag = input_bag[:-len(suffix)] + "_out" + suffix

    # 调用函数进行抽帧
    resync_bag_topics(input_bag, output_bag)
