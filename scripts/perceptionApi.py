#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import signal
import sys
import math

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_py.msg import RecognizedObjectArrayStamped
from detectron2_ros.msg import Result
from sympy import Point, Polygon, Line


class Perception:
    def __init__(self, useYolo, easyDetection, useFilteredObjects, classNameToBeDetected):
        # Variable Initialization
        self.img = None
        self.readObj = False
        self.pointingDirection = None
        self.pointingSlope = None
        self.pointingIntercept = None
        self.bridge = CvBridge()
        
        #
        self.useYolo = useYolo
        self.easyDetection = easyDetection
        self.useFilteredObjects = useFilteredObjects
        self.classNameToBeDetected = classNameToBeDetected

        # Variables
        self.detectedObjects = []
        self.filteredObjects = []

        # Msgs are defined in the mediapipeHolisticnode
        self.pointingLeftMsg = "left"
        self.pointingRightMsg = "right"

        # Topics
        if self.useYolo == True:
            self.camera_topic = "/object_detector/detection_image/compressed"
            self.readImgCompressed = True
            self.detectedObjects_topic = "/object_detector/detections"
        else:
            self.camera_topic = "/camera/color/image_raw"
            self.readImgCompressed = False
            self.detectedObjects_topic = "/detectron2_ros/result_yolo_msg"


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

        if self.detectedObjects == []:
            rospy.loginfo("No objects were detected!")

        if self.easyDetection:
            while(self.pointingDirection is None):
                rospy.loginfo("Getting pointing direction...")

            return  self.findObjectSimplifiedVersion()
        
        else:
            while(self.img is None):
                rospy.loginfo("Getting img...")

            while(self.pointingSlope is None and self.pointingIntercept is None):
                rospy.loginfo("Getting pointing slope and intercept...")

            res = self.lineIntersectionPolygon()
            if res != None:
                return res
            else:
                return self.findClosestObjectToLine()



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
        if self.useFilteredObjects:
            for obj in data.objects.objects:
                if obj.class_name == self.classNameToBeDetected:
                    self.detectedObjects.append(obj)
        else:
            for obj in data.objects.objects:    
                self.detectedObjects.append(obj)

        self.readObj = True

    
    def findObjectSimplifiedVersion(self):
        if self.pointingDirection != None:
            if self.pointingDirection == self.pointingLeftMsg:
                for idx, obj in enumerate(self.detectedObjects):
                    if idx == 0:
                        left_obj = obj

                    if obj.bounding_box.x_offset > left_obj.bounding_box.x_offset:
                        left_obj = obj

                return left_obj

            
            if self.pointingDirection == self.pointingRightMsg:
                for idx, obj in enumerate(self.detectedObjects):
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
            line = Line(Point(x1, y1), Point(x2, y2))
            for obj in self.detectedObjects:
                x_top_left = obj.bounding_box.x_offset
                y_top_left = obj.bounding_box.y_offset

                x_bottom_left = obj.bounding_box.x_offset
                y_bottom_left = obj.bounding_box.y_offset + obj.bounding_box.height

                x_top_right = obj.bounding_box.x_offset + obj.bounding_box.width
                y_top_right = obj.bounding_box.y_offset

                x_bottom_right = obj.bounding_box.x_offset + obj.bounding_box.width
                y_bottom_right = obj.bounding_box.y_offset + obj.bounding_box.height
        
                p1, p2, p3, p4, p5 = map(Point, [(x_bottom_left, y_bottom_left), (x_bottom_right, y_bottom_right), (x_top_right, y_top_right), (x_top_left, y_top_left), (x_bottom_left, y_bottom_left)])
                
                poly = Polygon(p1, p2, p3, p4, p5)

                isIntersection = poly.intersection(line)

                if isIntersection != []:
                    return obj

        return None


    def findClosestObjectToLine(self):
        returnObject = None
        returnDist = None
        
        if self.pointingSlope != None and self.pointingIntercept != None:
            perpendicularLineSlope = -1 / self.pointingSlope

            for idx, obj in enumerate(self.detectedObjects):
                # Find Bounding Box Center
                obj_boundingBoxCenter_x = obj.bounding_box.x_offset + obj.bounding_box.width / 2
                obj_boundingBoxCenter_y = obj.bounding_box.y_offset + obj.bounding_box.height / 2

                # Find Intercept of the perpendicular line
                perpendicularLineIntercept = obj_boundingBoxCenter_y - perpendicularLineSlope * obj_boundingBoxCenter_x

                # Find Intersection (Point)
                inter_x = (perpendicularLineIntercept - self.pointingIntercept) / (self.pointingSlope - perpendicularLineSlope)
                inter_y = self.pointingSlope * inter_x + self.pointingIntercept

                # Compute Distance between those two points (intersection and bounding box center)
                dx = math.pow(obj_boundingBoxCenter_x - inter_x, 2)
                dy = math.pow(obj_boundingBoxCenter_y - inter_y, 2)
                dist = math.sqrt(dx + dy)
                
                if idx == 0:
                    returnObject = obj
                    returnDist = dist
                
                if idx > 0 and returnDist > dist:
                    returnObject = obj
                    returnDist = dist

        return returnObject



def handler(signum, frame):
    exit(1)


def main():
    # Handle CTRL-C Interruption
    signal.signal(signal.SIGINT, handler)
    
    # Read Arguments
    yolo = True
    easyDetection = False
    useFilteredObjects = True
    classNameToBeDetected = "backpack"

    # if len(sys.argv) > 1 and sys.argv[1] == "detectron":
    #     yolo = False
    
    node_name = "perception_action"
    rospy.init_node(node_name, anonymous=True)
    rospy.loginfo("%s node created" % node_name)

    n_percep = Perception(yolo, easyDetection, useFilteredObjects, classNameToBeDetected)
    obj = n_percep.run()
    rospy.loginfo(obj)

    return obj


# Main function
if __name__ == '__main__':
    main()