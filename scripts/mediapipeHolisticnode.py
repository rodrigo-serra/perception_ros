#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import rospkg

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as imgPil
from facerecModule import *
from holisticDetectorModule import *
from PIL import ImageDraw
from scipy.spatial import ConvexHull
from perception_tests.msg import MediapipePointInfo, MediapipePointInfoArray



class MediapipeHolistic:
    def __init__(self):
        # Create the node
        node_name = "mediapipe_holistic"
        rospy.init_node(node_name, anonymous=False)
        rospy.loginfo("%s node created" % node_name)
        rospack = rospkg.RosPack()

        # Variable Initialization
        self.rate = rospy.Rate(10)
        self.img = None
        self.ctr = True
        self.detector = holisticDetector()
        self.currentEvent = "e_stop"
        self.bridge = CvBridge()
        self.directory = rospack.get_path('perception_tests')

        # Read from ROS Param
        self.camera_topic = rospy.get_param("~camera_topic")
        self.readImgCompressed = rospy.get_param("~img_compressed")
        self.usePointingHands = rospy.get_param("~pointing_hands")
        self.showImg = rospy.get_param("~visualization")
        self.drawPose = rospy.get_param("~drawPoseLandmarks")
        self.drawFace = rospy.get_param("~drawFaceLandmarks")
        self.drawRightHand = rospy.get_param("~drawRightHandLandmarks")
        self.drawLeftHand = rospy.get_param("~drawLeftHandLandmarks")
        self.drawFaceBoundary = rospy.get_param("~drawFaceBoundary")
        self.pointingRightHandMsg = rospy.get_param("~pointing_right_hand_msg")
        self.pointingLeftHandMsg = rospy.get_param("~pointing_left_hand_msg")

        # Subscribe to Camera Topic
        if self.readImgCompressed:
            self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
        else:
            self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)

        # Subscribe to Event and perform accordingly
        self.event_sub = rospy.Subscriber("~event_in", String, self.eventCallback)

        # Publish Face Landmarks
        self.mp_faceLandmarks_pub = rospy.Publisher("~face_landmarks", MediapipePointInfoArray, queue_size=10)
        
        # Publish Pose World Landmarks
        self.mp_poseWorldLandmarks_pub = rospy.Publisher("~pose_world_landmarks", MediapipePointInfoArray, queue_size=10)
        
        # Publish Img Pose Landmarks
        self.mp_imgPoseLandmarks_pub = rospy.Publisher("~img_pose_landmarks", MediapipePointInfoArray, queue_size=10)

        # Publish Right Hand Landmarks
        self.mp_rightHandLandmarks_pub = rospy.Publisher("~right_hand_landmarks", MediapipePointInfoArray, queue_size=10)

        # Publish Left Hand Landmarks
        self.mp_leftHandLandmarks_pub = rospy.Publisher("~left_hand_landmarks", MediapipePointInfoArray, queue_size=10)


        # EXTRAS
        # Publish Right Arm Length
        self.mp_rightArmLength_pub = rospy.Publisher("~right_arm_length", Float32, queue_size=10)
        
        # Publish Left Arm Length
        self.mp_leftArmLength_pub = rospy.Publisher("~left_arm_length", Float32, queue_size=10)
        
        # Publish Shoulder Length
        self.mp_shoulderLength_pub = rospy.Publisher("~shoulder_length", Float32, queue_size=10)

        # Publish Hip Length
        self.mp_hipLength_pub = rospy.Publisher("~hip_length", Float32, queue_size=10)

        # Publish Torso Length
        self.mp_torsoLength_pub = rospy.Publisher("~torso_length", Float32, queue_size=10)

        # Publish Hand Poiting Direction - Slope
        self.mp_pointingDirectionHand_slope_pub = rospy.Publisher("~hand_pointing_slope", Float32, queue_size=10)

        # Publish Hand Poiting Direction - Intercept
        self.mp_pointingDirectionHand_intercept_pub = rospy.Publisher("~hand_pointing_intercept", Float32, queue_size=10)

        # Publish Hand Poiting Direction - Direction
        self.mp_pointingDirectionHand_direction_pub = rospy.Publisher("~hand_pointing_direction", String, queue_size=10)
    
        # Publish Shirt/Sweater Color
        self.mp_sweaterColor_pub = rospy.Publisher("~sweater_color", String, queue_size=10)

    def run(self):
        while not rospy.is_shutdown():
            if self.currentEvent is not None:
                if self.currentEvent == "e_stop":
                    self.currentEvent = None
                    self.img = None
                    self.image_sub.unregister()
                    cv2.destroyAllWindows()
                    rospy.loginfo("Stopping detection!")

                if self.currentEvent == "e_start":
                    self.currentEvent = None
                    if self.readImgCompressed:
                        self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
                    else:
                        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)
                    rospy.loginfo("Starting detection!")

                if self.currentEvent == "e_reset":
                    self.img = None
                    self.ctr = True
                    self.detector = holisticDetector()
                    self.currentEvent = None
                    if self.readImgCompressed:
                        self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
                    else:
                        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)
                    rospy.loginfo("Reseting!")

            if self.img is not None:
                if self.ctr:
                    self.img = self.detector.find(self.img, self.drawPose, self.drawFace, self.drawRightHand, self.drawLeftHand)

                    isPoseWorldLandmarks = self.detector.getPoseWorldLandmarks()
                    if isPoseWorldLandmarks:
                        self.publishPoseWorldCoordinates()


                    isImgPoseLandmarks = self.detector.getPoseImgLandmarks(self.img)
                    if isImgPoseLandmarks:
                        self.publishPoseImgCoordinates()


                    isRightHandLandmarks = self.detector.getRightHandLandmarks(self.img)
                    if isRightHandLandmarks:
                        self.publishRightHandCoordinates()
                        
                    
                    isLeftHandLandmarks = self.detector.getLeftHandLandmarks(self.img)
                    if isLeftHandLandmarks:
                        self.publishLeftHandCoordinates()


                    if self.detector.getRightArmLength():
                        self.mp_rightArmLength_pub.publish(self.detector.getRightArmLength())

                    
                    if self.detector.getLeftArmLength():
                        self.mp_leftArmLength_pub.publish(self.detector.getLeftArmLength())

                    
                    if self.detector.getShoulderLength():
                        self.mp_shoulderLength_pub.publish(self.detector.getShoulderLength())

                    
                    if self.detector.getHipLength():
                        self.mp_hipLength_pub.publish(self.detector.getHipLength())


                    if self.detector.getTorsoLength():
                        self.mp_torsoLength_pub.publish(self.detector.getTorsoLength())


                    sweater_color = self.detector.readSweaterColor(self.img, self.directory)
                    if sweater_color:
                        self.mp_sweaterColor_pub.publish(sweater_color)


                    isPointingHand = self.detector.getPointingArm()
                    if isPointingHand:
                        if self.usePointingHands:
                            self.img, h_slope, h_intercept = self.detector.getPointingDirectionHand(self.img, isPointingHand)
                        else:
                            self.img, h_slope, h_intercept = self.detector.getPointingDirectionArm(self.img, isPointingHand)

                        if h_slope != None and h_intercept != None:
                            self.mp_pointingDirectionHand_slope_pub.publish(h_slope)
                            self.mp_pointingDirectionHand_intercept_pub.publish(h_intercept)
                            if h_slope > 0:
                                self.mp_pointingDirectionHand_direction_pub.publish(self.pointingLeftHandMsg)
                            else:
                                self.mp_pointingDirectionHand_direction_pub.publish(self.pointingRightHandMsg)


                    isFaceLandmarks = self.detector.getFaceLandmarks(self.img)
                    if isFaceLandmarks:
                        self.publishFaceCoordinates()
                        if self.drawFaceBoundary:
                            self.img = self.getFaceMask()
                    
                    
                    self.ctr = False
                    

                if self.showImg:
                    cv2.imshow("RealSense", self.img)
                    cv2.waitKey(1)                    
            
            self.rate.sleep()


        cv2.destroyAllWindows()
        rospy.loginfo('Shutting Down MediapipeHolistic Node')


    def getFaceMask(self):
        height, width, c = self.img.shape
        mask_img = imgPil.new('L', (width, height), 0)

        points = np.array(self.detector.faceCoordinates)

        hull = ConvexHull(points)
        polygon = []
        for v in hull.vertices:
            polygon.append((points[v, 0], points[v, 1]))

        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
        mask = np.array(mask_img)
        
        color_image = cv2.bitwise_and(self.img, self.img, mask=mask)

        return color_image


    def imgCallback(self, data):
        try:
            if self.readImgCompressed:
                self.img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            else:
                self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")

            self.ctr = True
        except CvBridgeError as e:
            print(e)


    def eventCallback(self, data):
        self.currentEvent = data.data

    
    def publishFaceCoordinates(self):
        msgArr = []
        for p in self.detector.faceCoordinates:
            msg = MediapipePointInfo()
            msg.x = p[0]
            msg.y = p[1]
            msg.z = -1
            msg.visibility = -1
            msgArr.append(msg)

        self.mp_faceLandmarks_pub.publish(msgArr)

    
    def publishPoseWorldCoordinates(self):
        msgArr = []
        for p in self.detector.poseCoordinates:
            msg = MediapipePointInfo()
            msg.x = p.x
            msg.y = p.y
            msg.z = p.z
            msg.visibility = p.visibility
            msgArr.append(msg)

        self.mp_poseWorldLandmarks_pub.publish(msgArr)

    
    def publishPoseImgCoordinates(self):
        msgArr = []
        for p in self.detector.imgPoseCoordinates:
            msg = MediapipePointInfo()
            msg.x = p.x
            msg.y = p.y
            msg.z = -1
            msg.visibility = p.visibility
            msgArr.append(msg)

        self.mp_imgPoseLandmarks_pub.publish(msgArr)


    def publishRightHandCoordinates(self):
        msgArr = []
        for p in self.detector.rightHandCoordinates:
            msg = MediapipePointInfo()
            msg.x = p[0]
            msg.y = p[1]
            msg.z = -1
            msg.visibility = -1
            msgArr.append(msg)

        self.mp_rightHandLandmarks_pub.publish(msgArr)


    def publishLeftHandCoordinates(self):
        msgArr = []
        for p in self.detector.leftHandCoordinates:
            msg = MediapipePointInfo()
            msg.x = p[0]
            msg.y = p[1]
            msg.z = -1
            msg.visibility = -1
            msgArr.append(msg)

        self.mp_leftHandLandmarks_pub.publish(msgArr)

        

        

# Main function
if __name__ == '__main__':
    camera = MediapipeHolistic()
    camera.run()