#!/usr/bin/env python3

import rospy
import cv2
import numpy as np

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_py.msg import RecognizedObjectArrayStamped

from shapely.geometry import Polygon, LineString


class Perception:
    def __init__(self):
        # Variable Initialization
        self.img = None
        self.detectedObjects = []
        self.filteredObjects = []
        self.readObj = False
        self.pointingDirection = None
        self.pointingSlope = None
        self.pointingIntercept = None

        self.easyDetection = False

        self.classNameToBeDetected = "backpack"

        self.pointingLeftMsg = "left"
        self.pointingRightMsg = "right"

        self.bridge = CvBridge()

        # Topics
        self.camera_topic = "/object_detector/detection_image/compressed"
        self.readImgCompressed = True

        self.detectedObjects_topic = "/object_detector/detections"
        
        self.pointingDirection_topic = "/perception/mediapipe_holistic/hand_pointing_direction"
        self.pointingSlope_topic = "/perception/mediapipe_holistic/hand_pointing_slope"
        self.pointingIntercept_topic = "/perception/mediapipe_holistic/hand_pointing_intercept"

        # Subscribe to Camera Topic
        if self.readImgCompressed:
            self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
        else:
            self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)

        
        self.objDetection_sub = rospy.Subscriber(self.detectedObjects_topic, RecognizedObjectArrayStamped, self.readDetectedObjects)

        self.pointingDirection_sub = rospy.Subscriber(self.pointingDirection_topic, String, self.getPointingDirection)

        self.pointingSlope_sub = rospy.Subscriber(self.pointingSlope_topic, Float32, self.getPointingSlope)

        self.pointingIntercept_sub = rospy.Subscriber(self.pointingIntercept_topic, Float32, self.getPointingIntercept)

    
    def run(self):
        while(not self.readObj):
            rospy.loginfo("Waiting for Object Detection...")


        if self.easyDetection:
            while(self.pointingDirection is None):
                rospy.loginfo("Getting pointing direction...")

            # if not self.filterObjectList():
            #     rospy.loginfo("No objects were detected with the follwing class: " + self.classNameToBeDetected)

            pointingObject = self.findObjectSimplifiedVersion()
            rospy.loginfo(pointingObject)
        
        else:
            if not self.filterObjectList():
                rospy.loginfo("No objects were detected with the follwing class: " + self.classNameToBeDetected)
            
            while(self.img is None):
                rospy.loginfo("Getting img...")

            while(self.pointingSlope is None and self.pointingIntercept is None):
                rospy.loginfo("Getting pointing slope and intercept...")


            self.lineIntersectionPolygon()





    def imgCallback(self, data):
        try:
            if self.readImgCompressed:
                self.img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            else:
                self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")


        except CvBridgeError as e:
            print(e)


    def getPointingDirection(self, data):
        self.pointingDirection = data.data

    
    def getPointingSlope(self, data):
        self.pointingSlope = data.data

    
    def getPointingIntercept(self, data):
        self.pointingIntercept = data.data


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

    
    def findObjectSimplifiedVersion(self):
        if self.pointingDirection != None:
            if self.pointingDirection == self.pointingLeftMsg:
                for idx, obj in enumerate(self.filteredObjects):
                    if idx == 0:
                        left_obj = obj

                    if obj.bounding_box.x_offset > left_obj.bounding_box.x_offset:
                        left_obj = obj

                return left_obj

            
            if self.pointingDirection == self.pointingRightMsg:
                for idx, obj in enumerate(self.filteredObjects):
                    if idx == 0:
                        right_obj = obj

                    if obj.bounding_box.x_offset < right_obj.bounding_box.x_offset:
                        right_obj = obj

                return right_obj

        return None


    def lineIntersectionPolygon(self):
        if self.pointingSlope != None and self.pointingIntercept != None:
            h, w, c = self.img.shape
            x1, x2 = 0, w
            y1, y2 = self.pointingIntercept * x1 + self.pointingIntercept, self.pointingSlope * x2 + self.pointingIntercept
            line1 = LineString([(y1, x1), (y2, x2)])
    
            for obj in self.filteredObjects:
                x_top_left = obj.bounding_box.x_offset
                y_top_left = obj.bounding_box.y_offset

                x_bottom_left = obj.bounding_box.x_offset
                y_bottom_left = obj.bounding_box.y_offset + obj.bounding_box.height

                x_top_right = obj.bounding_box.x_offset + obj.bounding_box.width
                y_top_right = obj.bounding_box.y_offset

                x_bottom_right = obj.bounding_box.x_offset + obj.bounding_box.width
                y_bottom_right = obj.bounding_box.y_offset + obj.bounding_box.height
        
                polygon = Polygon([(x_bottom_left, y_bottom_left), (x_bottom_right, y_bottom_right), (x_top_right, y_top_right), (x_top_left, y_top_left), (x_bottom_left, y_bottom_left)])

                rospy.logwarn("Bottom Left: " + str(x_bottom_left) + " " + str(y_bottom_left))
                rospy.logwarn("Bottom Right: " + str(x_bottom_right) + " " + str(y_bottom_right))
                rospy.logwarn("Top Right: " + str(x_top_right) + " " + str(y_top_right))
                rospy.logwarn("Top Left: " + str(x_top_left) + " " + str(y_top_left))

                res = line1.intersects(polygon)
                rospy.logerr(str(res))
                if res == True:
                    rospy.logwarn(res)
                    rospy.loginfo(obj)




        
        

        

# Main function
if __name__ == '__main__':
    node_name = "perception_action"
    rospy.init_node(node_name, anonymous=True)
    rospy.loginfo("%s node created" % node_name)
    n_percep = Perception()
    n_percep.run()