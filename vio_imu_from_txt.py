import rospy
from rospy import Time
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point
from sensor_msgs.msg import Imu
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
            data[0] =  float(data[0]) + 1719296183.905188096
            timestamp = Time.from_sec(data[0])
            imu_msg = Imu()
            imu_msg.header.stamp = timestamp
            imu_msg.angular_velocity.x = float(data[4])
            imu_msg.angular_velocity.y = float(data[5])
            imu_msg.angular_velocity.z = float(data[6])
            imu_msg.linear_acceleration.x = float(data[1])
            imu_msg.linear_acceleration.y = float(data[2])
            imu_msg.linear_acceleration.z = float(data[3])
            print(data)
            bag.write('/fvs/imu', imu_msg, Time.from_sec(data[0]-0.2))

        bag.close()
        print("Bag:{} 生成完成！".format(rosbag_file))

# 从命令行参数获取输入的txt文件名和输出的rosbag文件名
txt_file = sys.argv[1]
rosbag_name = sys.argv[2]
# 初始化ROS节点
rospy.init_node('rosbag_generator')

rosbag_file = rosbag_name + "_imu.bag"
# 调用函数生成ROS Bag文件
generate_rosbag(txt_file, rosbag_file)
