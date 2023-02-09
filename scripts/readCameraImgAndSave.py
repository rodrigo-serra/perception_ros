#!/usr/bin/env python3

import cv2
import rospy
import face_recognition
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import rospkg
import time
bridge = CvBridge()

class imageSubscriber(object):
    def __init__(self):
        node_name = "read_save_img"
        rospy.init_node(node_name, anonymous=False)
        rospy.loginfo("%s node created" % node_name)
        
        self.img = None
        self.readImgCompressed = False
        self.lastClockReading = None
        self.imgCounter = 0
        self.savingPeriod = 2
        self.saveDir = "/home/rodrigo/ros_ws/src/perception_tests/images/dataset/"

        self.topic_name = "/camera/color/image_raw"

        self.bridge = CvBridge()

        self.sub = rospy.Subscriber(self.topic_name, Image, self.imgCallback)


    def imgCallback(self, data):
        try:
            if self.readImgCompressed:
                self.img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            else:
                self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def run(self):
         while not rospy.is_shutdown():
            if self.img is not None:
                cv2.imshow("RealSense", self.img)
                cv2.waitKey(1)
                currentClockReading = time.process_time()
                if self.lastClockReading == None or currentClockReading - self.lastClockReading > self.savingPeriod:
                    rospy.logwarn("SAVING IMG!")
                    cv2.imwrite(self.saveDir + "img_" + str(self.imgCounter) + ".png", self.img)
                    self.imgCounter += 1
                    self.lastClockReading = currentClockReading

def main():
    c = imageSubscriber()
    c.run()

if __name__ == '__main__':
    main()