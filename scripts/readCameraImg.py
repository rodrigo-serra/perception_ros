#!/usr/bin/env python

import cv2
import rospy
import face_recognition
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospkg

# get an instance of RosPack with the default search paths
# rospack = rospkg.RosPack()

# get the file path for face_recognition_pkg
# path_to_package = rospack.get_path("perception_tests")
# print(path_to_package)

bridge = CvBridge()

class imageSubscriber(object):
    def __init__(self, topic_name):
        self.sub = rospy.Subscriber(topic_name, Image, self.imgCallback)


    def imgCallback(self, topic_data):
        # try:
        cv_img = bridge.imgmsg_to_cv2(topic_data)
        # except CvBridgeError:
        #     rospy.logerr("CvBridge Error")

        # self.show_image(cv_img)
        cv2.imshow("Image Window", cv_img)
        cv2.waitKey(1)


    def show_image(self, img):
        cv2.imshow("Image Window", img)
        cv2.waitKey(1)



if __name__ == '__main__':
    rospy.init_node('read_image')
    # Initialize an OpenCV Window named "Image Window"
    cv2.namedWindow("Image Window", 1)
    imgSubscriber = imageSubscriber("/camera/color/image_raw")
    rospy.spin()
    cv2.destroyAllWindows()