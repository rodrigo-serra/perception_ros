#!/usr/bin/env python3

import rospy
import cv2
import numpy as np

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_py.msg import RecognizedObjectArrayStamped



class Perception:
    def __init__(self):
        # Variable Initialization
        self.img = None
        self.detectedObjects = []
        self.filteredObjects = []
        self.readObj = False
        self.readImg = False
        self.pointingDirection = None

        self.classNameToBeDetected = "backpack"

        self.pointingLeftMsg = "left"
        self.pointingRightMsg = "right"

        self.bridge = CvBridge()

        # Read from ROS Param
        self.camera_topic = "/object_detector/detection_image/compressed"
        self.readImgCompressed = True
        self.detectedObjects_topic = "/object_detector/detections"
        self.pointingDirection_topic = "/perception/mediapipe_holistic/hand_pointing_direction"

        # Subscribe to Camera Topic
        if self.readImgCompressed:
            self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
        else:
            self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)

        
        self.objDetection_sub = rospy.Subscriber(self.detectedObjects_topic, RecognizedObjectArrayStamped, self.readDetectedObjects)

        self.pointingDirection_sub = rospy.Subscriber(self.pointingDirection_topic, String, self.getPointingDirection)

    
    def run(self):
        while(not self.readObj and not self.readImg):
            rospy.loginfo("Waiting for Object Detection...")

        if self.filterObjectList():
            rospy.loginfo(self.filteredObjects)
        else:
            rospy.loginfo("No objects were detected with the follwing class: " + self.classNameToBeDetected)


    def imgCallback(self, data):
        try:
            if self.readImgCompressed:
                self.img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            else:
                self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")

            self.readImg = True

        except CvBridgeError as e:
            print(e)


    def getPointingDirection(self, data):
        self.pointingDirection = data.data


    def readDetectedObjects(self, data):
        for obj in data.objects.objects:
            self.detectedObjects.append(obj)

        self.readObj = True

    
    def filterObjectList(self):
        if self.detectedObjects != []:
            for obj in self.detectedObjects:
                if obj.class_name == self.classNameToBeDetected:
                    self.filteredObjects.append(obj)
            return True
        return False

    
    # def findObjectSimplifiedVersion(self):
    #     if self.pointingDirection != None:
    #         if self.pointingDirection == self.pointingLeftMsg:
    #             h, w, c = self.img.shape
    #             for obj in self.filteredObjects:
    #                 if obj.bounding_box.x_offset:





        
        

        

# Main function
if __name__ == '__main__':
    node_name = "perception_action"
    rospy.init_node(node_name, anonymous=True)
    rospy.loginfo("%s node created" % node_name)
    n_percep = Perception()
    n_percep.run()