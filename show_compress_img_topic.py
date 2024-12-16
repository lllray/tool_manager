import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import sys
import numpy as np
def image_callback(msg):
    try:
        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imshow("Compressed Image", image_np)
        cv2.waitKey(1)  # Refresh the display
    except Exception as e:
        print(e)

def image_listener(topic):
    rospy.init_node('image_listener', anonymous=True)
    rospy.Subscriber(topic, CompressedImage, image_callback)
    rospy.spin()

if __name__ == '__main__':
    topic = sys.argv[1]
    bridge = CvBridge()
    image_listener(topic)
