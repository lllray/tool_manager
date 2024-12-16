import rospy
from rospy import Time
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point
import rosbag
import sys

nav_start_time = 0
ros_start_time = 0

def generate_rosbag(txt_file, rosbag_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        bag = rosbag.Bag(rosbag_file, 'w')
        for line in lines:
            data = line.split()
            data[0] = float(data[0])
            timestamp = Time.from_sec(data[0])
            range_data = Range()
            range_data.header.stamp = timestamp
            range_data.header.frame_id = 'sensor_frame'  # 设置frame_id
            range_data.radiation_type = Range.INFRARED
            range_data.field_of_view = 0.1  # 设置视场角
            range_data.min_range = 0.0  # 设置最小测量距离
            range_data.max_range = 10.0  # 设置最大测量距离
            range_data.range = float(data[3])  # 第4列为对地面高度

            bag.write('/distance', range_data, timestamp)

        bag.close()
        print("Bag:{} 生成完成！".format(rosbag_file))

# 从命令行参数获取输入的txt文件名和输出的rosbag文件名
txt_file = sys.argv[1]
rosbag_name = sys.argv[2]
# 初始化ROS节点
rospy.init_node('rosbag_generator')

rosbag_file = rosbag_name + "_tof.bag"
# 调用函数生成ROS Bag文件
generate_rosbag(txt_file, rosbag_file)
