import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from copy import deepcopy
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3



class VideoRecording(Node):

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        super().__init__('video_recording')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.count = 0

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv2.namedWindow('video_window')
        while True:
            self.run_loop()
            time.sleep(0.1)



    def run_loop(self):
        if not self.cv_image is None:
            cv2.imshow('video_window', self.cv_image)
            cv2.imwrite("/home/alexiswu/ros2_ws/src/compRobo22_CV/hand_gestures/stop/frame_%d.jpg" %self.count,self.cv_image)
            self.count += 1
            cv2.waitKey(5)


if __name__ == '__main__':
    node = VideoRecording("/camera/image_raw")
    node.run()


def main(args=None):
    rclpy.init()
    n = VideoRecording("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()