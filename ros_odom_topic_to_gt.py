import rosbag
import sys

bag_file = sys.argv[1]
topic_name = sys.argv[2]

suffix = ".bag"
output_file = bag_file[:-len(suffix)] + "_gt.txt"

with open(output_file, 'w') as file:
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages(topics=[topic_name]):  # 替换为你的 odometry 话题名称
        if msg._type == 'nav_msgs/Odometry':
            timestamp = msg.header.stamp.to_sec()
            position = msg.pose.pose.position

            file.write(f'{timestamp} {position.x} {position.y} {position.z} 0 0 0 1\n')
            print(f'{timestamp} {position.x} {position.y} {position.z} 0 0 0 1\n')

    bag.close()

print("Data extraction completed.")
