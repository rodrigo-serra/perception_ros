#!/usr/bin/env python3

import rospy, rospkg
import cv2
import numpy as np
import math
import shutil, os, gdown
from io import BytesIO
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from detectron2_ros.msg import Result, RecognizedObjectArrayStamped, RecognizedObjectWithMaskArrayStamped, SingleRecognizedObjectWithMask
from perception_tests.msg import MediapipePointInfo, MediapipePointInfoArray
from perception_tests.msg import ReidInfoArray
from sympy import Point, Polygon, Line

import message_filters

# import tiago_object_localization.tiago_object_localization_library_helper as obj_pose_module

# from mbot_perception_msgs.msg import TrackedObject3DList, TrackedObject3D, RecognizedObject3DList, RecognizedObject3D
# from mbot_perception_msgs.srv import DeleteObject3D, DeleteObject3DRequest

class Perception:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # self.tracked_objects=None
        # self.subscriber = rospy.Subscriber("/bayes_objects_tracker/tracked_objects", TrackedObject3DList, self.__trackCallback)

        # Variable Initialization
        self.__timeout = 3
        self.__img = None
        
        self.__pointingDirection = None
        self.__pointingSlope = None
        self.__pointingIntercept = None
        
        self.__faceLandmarks = None
        self.__poseWorldLandmarks = None
        self.__imgPoseLandmarks = None
        self.__rightHandLandmarks = None
        self.__leftHandLandmarks = None
        
        self.__hipLength = None
        self.__shoulderLength = None
        self.__torsoLength = None
        self.__rightArmLength = None
        self.__leftArmLength = None

        self.__sweaterColor = None

        self.__peopleDetection = None
        self.__peopleDetectionRecord = None
        
        self.__detectedObjects = []
        self.__detectionMsg = None
        self.__depthImg = None
        self.__readDetectronMsgs = False


    def is_door_open(self, timeout=3.0):
        """
        description: assuming that there is a door in front of the robot and using the robot laser scanners detect if its open or closed
        input: None
        output (return): A boolean with the status of the door, true: door is open, false: door is closed
        NOTE: raises exceptions in some cases to prevent mistaking with boolean door open/close
        """

        self.__door_detector_pub = rospy.Publisher('/door_detector_node/event_in', String, queue_size=5)
        rospy.sleep(0.25)
        sub_door = rospy.Subscriber('/door_detector_node/event_out', String, self.__callback_door_open)
        rospy.sleep(0.25)

        if self.__door_detector_pub.get_num_connections() == 0:
            rospy.logwarn('There is no subscriber for door detector topic. Did you launch the door detector?')

        self.__door_open = None
        self.__door_detector_pub.publish(String(data='e_start'))

        # wait for response for 3 seconds
        wait_time = rospy.Time.now() + rospy.Duration(timeout)
        while self.__door_open is None:
            if rospy.Time.now() > wait_time:
                rospy.logerr("Timeout while checking door")
                return None

        sub_door.unregister()

        return self.__door_open


    def __callback_door_open(self, msg):
        if msg.data == 'e_open':
            self.__door_open = True
        elif msg.data == 'e_closed':
            self.__door_open = False

    
    def downloadDetectronModel(self, option):
        rospack = rospkg.RosPack()
        detectronDir = rospack.get_path('detectron2_ros')
        detectronModelsDir = detectronDir + "/model"
        modelZipfile = detectronDir + "/custom_model.zip"

        # Delete model folder
        if os.path.exists(detectronModelsDir):
            shutil.rmtree(detectronModelsDir, ignore_errors=True)

        if option == 1:
            # Download Fashion Model
            pass
        elif option == 2:
            # Download Bags/BagsHandles Model
            pass
        elif option == 3:
            # Download DoorKnobs Model
            # url = "https://drive.google.com/file/d/1l83vq4ybTUpuDwXrSMRqQJzrjfvfZoKh/view?usp=sharing"
            url = "https://ulisboa-my.sharepoint.com/:u:/g/personal/ist181272_tecnico_ulisboa_pt/ES2K9nbG4EVOvUG4RuNV010BQQhWVS4gwkt7VJmCsGCBqA?e=hVAaXd"
        else:
            rospy.logwarn("Option is not available!")
            return

        # Download model under zip format
        rospy.logwarn("Downloading Model zip file!")
        
        
        # try:
        #     gdown.download(url, modelZipfile, quiet=False,fuzzy=True)
        # except:
        #     rospy.logwarn("Could not download the model!")
            # return
        
        rospy.sleep(3)
        
        # Unzip file
        # rospy.logwarn("Unzipping Model!")
        # try:
        #     shutil.unpack_archive(modelZipfile , detectronDir)
        # except:
        #     rospy.logwarn("Could not unzip the model")
        
        # os.remove(modelZipfile)


    def detectPointingObject(self, classNameToBeDetected, useYolo = False, easyDetection = False, useFilteredObjects = True, score = 0.5):
        """
        This action returns the object someone is pointing at. It requires the mediapipe holistic node to be running and the Detectron or YOLO nodes. The msg type is RecognizedObject.
        
        :param useYolo: (bool) It tells which object detector should we subsribe to. If set to True, subscribes to YOLO otherwise uses Detectron.
        :param easyDetection: (bool) If set to true, it focuses on the arm direction to determine the pointing direction. Then it selects the object farthest left or farthest right accordingly. 
                                    This approach works under the assumption that the useFilteredObjects is also set to true and that we are choosing between two objects. 
                                    If set to false, it finds which object gets intercepted by the pointing line segment and returns that object. If no object is detected, it returns the one closest to the line.
        :param classNameToBeDetected: (string) The class name to be filtered during the search.
        :param score: (float) The detection confindence level.
        
        :return res: (RecognizedObject.mgs) It returns the object.
        """ 
        if useFilteredObjects == True:
            if type(classNameToBeDetected) != list:
                rospy.logwarn("Input argument classNameToBeDetected must be a list of classes!")
                return None
            
            if len(classNameToBeDetected) == 0:
                rospy.logwarn("Input argument classNameToBeDetected cannot be an empty list!")
                return None
                
        self.__detectionMsg = self.returnDetectedObjects()
        if self.__detectionMsg is None:
            return None

        self.__detectedObjects = self.__filterObjectionDetectionMsg(self.__detectionMsg, useFilteredObjects, classNameToBeDetected, score)
        
        if self.__detectedObjects == []:
            return None

        return self.__returnPointedObject(easyDetection, useYolo)
        

    def detectPointingObjectWithCustomMsg(self, classNameToBeDetected, easyDetection = False, useFilteredObjects = True, score = 0.5):
        """
        This action returns the object someone is pointing at + the corresponding depth img. It requires the mediapipe holistic node to be running and the Detectron node. The msg type is SingleRecognizedObjectWithMask.
        
        :param easyDetection: (bool) If set to true, it focuses on the arm direction to determine the pointing direction. Then it selects the object farthest left or farthest right accordingly. 
                                    This approach works under the assumption that the useFilteredObjects is also set to true and that we are choosing between two objects. 
                                    If set to false, it finds which object gets intercepted by the pointing line segment and returns that object. If no object is detected, it returns the one closest to the line.
        :param classNameToBeDetected: (string) The class name to be filtered during the search.
        :param score: (float) The detection confindence level.
        
        :return res: (SingleRecognizedObjectWithMask.mgs + Image.msg) It returns the object and the corresponding depth image.
        """
        if useFilteredObjects == True:
            if type(classNameToBeDetected) != list:
                rospy.logwarn("Input argument classNameToBeDetected must be a list of classes!")
                return None, None
            
            if len(classNameToBeDetected) == 0:
                rospy.logwarn("Input argument classNameToBeDetected cannot be an empty list!")
                return None, None

        useYolo = False
        try:
            readDetectronCustomMsg = rospy.get_param("/detectron2_ros/use_detectron_custom_msg")
        except:
            rospy.logwarn("Detectron custom msg must be set to true on the detectron launch file!")
            return None, None

        self.__readSynchronizedMsgs()
        if self.__detectionMsg is None:
            rospy.logwarn("Could not read detectron detection and image depth topics!")
            return None, None

        self.__detectedObjects = self.__filterObjectionDetectionMsg(self.__detectionMsg, useFilteredObjects, classNameToBeDetected, score)
        
        if self.__detectedObjects == []:
            return None, None

        obj = self.__returnPointedObject(easyDetection, useYolo)
        if obj == None:
            return None, None

        msg = SingleRecognizedObjectWithMask()
        msg.header = self.__detectionMsg.header
        msg.object = obj

        return msg, self.__depthImg


    def returnDetectedObjects(self, useYolo = False, useFilteredObjects = True, classNameToBeDetected = 'bag', score = 0.5):
        """
        It returns all the objects detected by the YOLO or the detectron. Hence, it requires one of the nodes to be running. The msg type is RecognizedObjectArrayStamped.
        
        :param useYolo: (bool) It tells which object detector should we subsribe to. If set to True, subscribes to YOLO otherwise uses Detectron.
        :param classNameToBeDetected: (string) The class name to be filtered during the search.
        :param score: (float) The detection confindence level.
        
        :return dObjects: (RecognizedObjectArrayStamped.mgs) It returns all objects detected.
        """ 
        
        detectedObjects_topic = "/detectron2_ros/result_yolo_msg"
        if useYolo == True:
            detectedObjects_topic = "/object_detector/detections"

        try:
            data = rospy.wait_for_message(detectedObjects_topic, RecognizedObjectArrayStamped, timeout = self.__timeout)
        except:
            rospy.logerr("Object Detection Results are not being published!")
            return None

        return data


    def __returnPointedObject(self, easyDetection, useYolo):
        """
        It returns the object pointed at.
        
        :param easyDetection: (bool) If set to true, it focuses on the arm direction to determine the pointing direction. Then it selects the object farthest left or farthest right accordingly. 
                                    This approach works under the assumption that the useFilteredObjects is also set to true and that we are choosing between two objects. 
                                    If set to false, it finds which object gets intercepted by the pointing line segment and returns that object. If no object is detected, it returns the one closest to the line.
        :param useYolo: (bool) In this case it tells which img topic should we subscribe to.
        
        :return object: It returns the object pointed at. The msg type can be either RecognizedObject.msg or RecognizedObjectWithMask.msg
        """ 
        if easyDetection:
            self.getPointingDirection()
            return self.__findObjectSimplifiedVersion()
        else:
            self.__getImg(useYolo)
            if self.__img is None:
                return None

            self.getPointingSlope()
            self.getPointingIntercept()
            
            res = self.__lineIntersectionPolygon()
            if res != None:
                return res
            else:
                return self.__findClosestObjectToLine()


    def __filterObjectionDetectionMsg(self, data, useFilteredObjects, classNameToBeDetected, score):
        """
        It returns the list of detected objects.
        
        :param easyDetection: (bool) If set to true, it focuses on the arm direction to determine the pointing direction. Then it selects the object farthest left or farthest right accordingly. 
                                    This approach works under the assumption that the useFilteredObjects is also set to true and that we are choosing between two objects. 
                                    If set to false, it finds which object gets intercepted by the pointing line segment and returns that object. If no object is detected, it returns the one closest to the line.
        :param useYolo: (bool) In this case it tells which img topic should we subscribe to.
        
        :return object: It returns the object pointed at. The msg type can be either RecognizedObject.msg or RecognizedObjectWithMask.msg
        """ 
        dObjects = []
        if useFilteredObjects:
            for obj in data.objects.objects:
                if obj.class_name in classNameToBeDetected and obj.confidence > score:
                    dObjects.append(obj)
        else:
            for obj in data.objects.objects:    
                dObjects.append(obj)

        return dObjects


    def __detectronSynchronizedCallback(self, detectronMsg, depthImgMsg):
        self.__detectionMsg = detectronMsg
        self.__depthImg = depthImgMsg
        self.__readDetectronMsgs = True
    

    def __readSynchronizedMsgs(self):
        detectronMsgObjects_topic = "/detectron2_ros/result"
        depthImg_topic = "/camera/aligned_depth_to_color/image_raw"

        detectron_sub = message_filters.Subscriber(detectronMsgObjects_topic, RecognizedObjectWithMaskArrayStamped)
        depth_img_sub = message_filters.Subscriber(depthImg_topic, Image)

        ts = message_filters.TimeSynchronizer([detectron_sub, depth_img_sub], 10)
        ts.registerCallback(self.__detectronSynchronizedCallback)
        rospy.sleep(5)

        while not self.__readDetectronMsgs:
            break

        self.__readDetectronMsgs = False          

    
    def returnDetectedObjectsDetectronMsg(self):
        detectronMsgDetectedObjects_topic = "/detectron2_ros/result"

        try:
            data = rospy.wait_for_message(detectronMsgDetectedObjects_topic, Result, timeout = self.__timeout)
        except:
            rospy.logerr("Could read detectron msg!")
            return None

        return data


    def getObjectNames(self):
        """
        It returns all the objects class names detected by the YOLO or the detectron. Hence, it requires one of the nodes to be running.
    
        :return objs_class_names: (list) It returns a list of string with all objects class names detected.
        """ 
        data = self.returnDetectedObjects(useYolo = False, useFilteredObjects = False)
        if data is None:
            return None

        objs = self.__filterObjectionDetectionMsg(data, False, None, None)
        
        if len(objs) == 0:
            rospy.logerr("Could not get objects!")
            return
        else:
            objs_class_names = []
            for obj in objs:
                objs_class_names.append(obj.class_name)
            return objs_class_names


    def __getImg(self, useYolo):
        """
        It reads an image from a specified topic.
        
        :param useYolo: (bool) It tells which topic should we subsribe to. If set to True, subscribes to YOLO image otherwise uses the camera image.
        
        """ 
        bridge = CvBridge()

        if useYolo == True:
            camera_topic = "/object_detector/detection_image/compressed"
            readImgMsg = CompressedImage
        else:
            camera_topic = "/camera/color/image_raw"
            readImgMsg = Image

        try:
            data = rospy.wait_for_message(camera_topic, readImgMsg, timeout = self.__timeout)
        except:
            rospy.logerr("Could not read img from: " + camera_topic)
            return

        try:
            if readImgMsg == CompressedImage:
                self.__img = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            else:
                self.__img = bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)


    def getPointingDirection(self):
        pointingDirection_topic = "/perception/mediapipe_holistic/hand_pointing_direction"
        try:
            data = rospy.wait_for_message(pointingDirection_topic, String, timeout = self.__timeout)
            self.__pointingDirection = data.data
            return self.__pointingDirection
        except:
            rospy.logerr("Could not get Pointing Direction!")


    
    def getPointingSlope(self):
        pointingSlope_topic = "/perception/mediapipe_holistic/hand_pointing_slope"
        try:
            data = rospy.wait_for_message(pointingSlope_topic, Float32, timeout = self.__timeout)
            self.__pointingSlope = data.data
            return self.__pointingSlope
        except:
            rospy.logerr("Could not get Slope of the Pointing Line Segment!")

    

    def getPointingIntercept(self):
        pointingIntercept_topic = "/perception/mediapipe_holistic/hand_pointing_intercept"
        try:
            data = rospy.wait_for_message(pointingIntercept_topic, Float32, timeout = self.__timeout)
            self.__pointingIntercept = data.data
            return self.__pointingIntercept
        except:
            rospy.logerr("Could not get Intercept of the Pointing Line Segment!")


    
    def __findObjectSimplifiedVersion(self):
        # Msgs are defined in the mediapipeHolisticnode launch file
        try:
            pointingLeftMsg = rospy.get_param("/perception/mediapipe_holistic/pointing_left_hand_msg")
            pointingRightMsg = rospy.get_param("/perception/mediapipe_holistic/pointing_right_hand_msg")
        except:
            rospy.logerr ("Mediapipe node must be running!")
            return None

        
        if self.__pointingDirection != None:
            if self.__pointingDirection == pointingLeftMsg:
                for idx, obj in enumerate(self.__detectedObjects):
                    if idx == 0:
                        left_obj = obj

                    if obj.bounding_box.x_offset > left_obj.bounding_box.x_offset:
                        left_obj = obj

                return left_obj

            
            if self.__pointingDirection == pointingRightMsg:
                for idx, obj in enumerate(self.__detectedObjects):
                    if idx == 0:
                        right_obj = obj

                    if obj.bounding_box.x_offset < right_obj.bounding_box.x_offset:
                        right_obj = obj

                return right_obj

        return None


    def __lineIntersectionPolygon(self):
        if self.__pointingSlope != None and self.__pointingIntercept != None:
            h, w, c = self.__img.shape
            x1, x2 = 0, w
            y1, y2 = self.__pointingIntercept * x1 + self.__pointingIntercept, self.__pointingSlope * x2 + self.__pointingIntercept
            line = Line(Point(x1, y1), Point(x2, y2))

            for obj in self.__detectedObjects:
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


    def __findClosestObjectToLine(self):
        returnObject = None
        returnDist = None
        
        if self.__pointingSlope != None and self.__pointingIntercept != None:
            perpendicularLineSlope = -1 / self.__pointingSlope

            for idx, obj in enumerate(self.__detectedObjects):
                # Find Bounding Box Center
                obj_boundingBoxCenter_x = obj.bounding_box.x_offset + obj.bounding_box.width / 2
                obj_boundingBoxCenter_y = obj.bounding_box.y_offset + obj.bounding_box.height / 2

                # Find Intercept of the perpendicular line
                perpendicularLineIntercept = obj_boundingBoxCenter_y - perpendicularLineSlope * obj_boundingBoxCenter_x

                # Find Intersection (Point)
                inter_x = (perpendicularLineIntercept - self.__pointingIntercept) / (self.__pointingSlope - perpendicularLineSlope)
                inter_y = self.__pointingSlope * inter_x + self.__pointingIntercept

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


    def getPoseWorldLandmarks(self):
        poseWorldLandmarks_topic = "/perception/mediapipe_holistic/pose_world_landmarks"
        try:
            data = rospy.wait_for_message(poseWorldLandmarks_topic, MediapipePointInfoArray, timeout = self.__timeout)
            self.__poseWorldLandmarks = data
            return self.__poseWorldLandmarks
        except:
            rospy.logerr("Could not get Pose World Landmarks!")



    def getFaceLandmarks(self):
        faceLandmarks_topic = "/perception/mediapipe_holistic/face_landmarks"
        try:
            data = rospy.wait_for_message(faceLandmarks_topic, MediapipePointInfoArray, timeout = self.__timeout)
            self.__faceLandmarks = data
            return self.__faceLandmarks
        except:
            rospy.logerr("Could not get Face Landmarks!")



    def getImgPoseLandmarks(self):
        imgPoseLandmarks_topic = "/perception/mediapipe_holistic/img_pose_landmarks"
        try:
            data = rospy.wait_for_message(imgPoseLandmarks_topic, MediapipePointInfoArray, timeout = self.__timeout)
            self.__imgPoseLandmarks = data
            return self.__imgPoseLandmarks
        except:
            rospy.logerr("Could not get Img Pose Landmarks!")


    
    def getRightHandLandmarks(self):
        rightHandLandmarks_topic = "/perception/mediapipe_holistic/right_hand_landmarks"
        try:
            data = rospy.wait_for_message(rightHandLandmarks_topic, MediapipePointInfoArray, timeout = self.__timeout)
            self.__rightHandLandmarks = data
            return self.__rightHandLandmarks
        except:
            rospy.logerr("Could not get Right Hand Landmarks!")



    def getLeftHandLandmarks(self):
        leftHandLandmarks_topic = "/perception/mediapipe_holistic/left_hand_landmarks"
        try:
            data = rospy.wait_for_message(leftHandLandmarks_topic, MediapipePointInfoArray, timeout = self.__timeout)
            self.__leftHandLandmarks = data
            return self.__leftHandLandmarks
        except:
            rospy.logerr("Could not get Left Hand Landmarks!")



    def getHipLength(self):
        hipLength_topic = "/perception/mediapipe_holistic/hip_length"
        try:
            data = rospy.wait_for_message(hipLength_topic, Float32, timeout = self.__timeout)
            self.__hipLength = data.data
            return self.__hipLength
        except:
            rospy.logerr("Could not get Hip Length!")


    
    def getTorsoLength(self):
        torsoLength_topic = "/perception/mediapipe_holistic/torso_length"
        try:
            data = rospy.wait_for_message(torsoLength_topic, Float32, timeout = self.__timeout)
            self.__torsoLength = data.data
            return self.__torsoLength
        except:
            rospy.logerr("Could not get Torso Length!")

   
   
    def getShoulderLength(self):
        shoulderLength_topic = "/perception/mediapipe_holistic/shoulder_length"
        try:
            data = rospy.wait_for_message(shoulderLength_topic, Float32, timeout = self.__timeout)
            self.__shoulderLength = data.data
            return self.__shoulderLength
        except:
            rospy.logerr("Could not get Shoulder Length!")

    
    
    def getRightArmLength(self):
        rightArmLength_topic = "/perception/mediapipe_holistic/right_arm_length"
        try:
            data = rospy.wait_for_message(rightArmLength_topic, Float32, timeout = self.__timeout)
            self.__rigthArmLength = data.data
            return self.__rigthArmLength
        except:
            rospy.logerr("Could not get Right Arm Length!")

    
    
    def getLeftArmLength(self):
        leftArmLength_topic = "/perception/mediapipe_holistic/left_arm_length"
        try:
            data = rospy.wait_for_message(leftArmLength_topic, Float32, timeout = self.__timeout)
            self.__leftArmLength = data.data
            return self.__leftArmLength
        except:
            rospy.logerr("Could not get Left Arm Length!")



    def getPeopleDetection(self):
        peopleDetection_topic = "/perception/reid/current_detection"
        try:
            data = rospy.wait_for_message(peopleDetection_topic, ReidInfoArray, timeout = self.__timeout)
            self.__peopleDetection = data.reidArr
            return self.__peopleDetection
        except:
            rospy.logerr("Could not get People Detection!")

    
    
    def getPeopleDetectionRecord(self):
        peopleDetectionRecord_topic = "/perception/reid/detection_record"
        try:
            data = rospy.wait_for_message(peopleDetectionRecord_topic, ReidInfoArray, timeout = self.__timeout)
            self.__peopleDetectionRecord = data.reidArr
            return self.__peopleDetectionRecord
        except:
            rospy.logerr("Could not get People Detection Record!")


    def getClosestPersonToCamera(self):
        detections = self.getPeopleDetection()
        if detections is None:
            rospy.logwarn("Currently not detecting anyone")
            return None

        person_idx = -1
        current_area = -1

        for idx, d in enumerate(detections):
            bounding_box_area = (d.right - d.left) * (d.bottom - d.top)
            if bounding_box_area > current_area:
                current_area = bounding_box_area
                person_idx = idx

        return detections[person_idx]        
            
    
    def readSweaterColor(self):
        sweaterColor_topic = "/perception/mediapipe_holistic/sweater_color"
        try:
            data = rospy.wait_for_message(sweaterColor_topic, String, timeout = self.__timeout)
            self.__sweaterColor = data.data
            return self.__sweaterColor
        except:
            rospy.logerr("Could not get Sweater/T-shirt color!")


    def ___eventIn(self, msg, option):
        publishMsg = True
        if option == "reid":
            pub = rospy.Publisher("/perception/reid/event_in", String, queue_size=10)
        elif option == "mediapipe_holistic":
            pub = rospy.Publisher("/perception/mediapipe_holistic/event_in", String, queue_size=10)
        elif option == "detectron":
            pub = rospy.Publisher("/detectron2_ros/event_in", String, queue_size=10)
        else:
            publishMsg = False    
        
        rospy.sleep(1)
        
        if publishMsg:
            e = String()
            e.data = msg
            pub.publish(e)
    
    
    def startReid(self):
        self.___eventIn("e_start", "reid")
    

    def stopReid(self):
        self.___eventIn("e_stop", "reid")
    
    
    def resetReid(self):
        self.___eventIn("e_reset", "reid")
    
    
    def takePhotoReid(self):
        self.___eventIn("e_take_photo", "reid")
    
    
    def enableAutomaticReid(self):
        self.___eventIn("e_enable_automatic", "reid")
    
    
    def disableAutomaticReid(self):
        self.___eventIn("e_disable_automatic", "reid")


    def startMediapipeHolistic(self):
        self.___eventIn("e_start", "mediapipe_holistic")

    
    def stopMediapipeHolistic(self):
        self.___eventIn("e_stop", "mediapipe_holistic")
    
    
    def resetMediapipeHolistic(self):
        self.___eventIn("e_reset", "mediapipe_holistic")


    def startDetectron(self):    
        self.___eventIn("e_start", "detectron")
    

    def startDetectronTopics(self):    
        self.___eventIn("e_start_topics", "detectron")


    def stopDetectron(self):    
        self.___eventIn("e_stop", "detectron")
    

    def stopDetectronTopics(self):    
        self.___eventIn("e_stop_topics", "detectron")

    # def get_object_pose(self, pointing_object, depth_frame=None):
    #     if pointing_object == None:
    #         rospy.logerr('Inputed pointing_object is None, please input a valid object.')
    #         return None
    #     if not depth_frame:
    #         try:
    #             depth_frame = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image, timeout = self.__timeout)
    #         except:
    #             rospy.logerr("Could not get depth frame!")
        
    #     CameraIntrinsics = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo, timeout = self.__timeout)

    #     #camera intrinsics initialization
    #     fx_d = CameraIntrinsics.K[0]
    #     fy_d = CameraIntrinsics.K[4]
    #     cx_d = CameraIntrinsics.K[2]
    #     cy_d = CameraIntrinsics.K[5]
    #     depthScale = 1000
        
    #     #converting msg to image - getting mask and depth image
    #     mask = obj_pose_module.convert_to_cv_image(pointing_object.object.mask)
    #     image_depth = obj_pose_module.convert_to_cv_image(depth_frame)

    #     #getting bounding box limits
    #     x_offset = pointing_object.object.bounding_box.x_offset
    #     width = pointing_object.object.bounding_box.width
    #     y_offset = pointing_object.object.bounding_box.y_offset
    #     height = pointing_object.object.bounding_box.height

    #     #get pose
    #     half_cube_dimension = 5 #sets dimension of cube from which the depth mean is got - cube dimension is set to (2*half_cube_dimension + 1)
    #     x, y, z = obj_pose_module.get_center_mask(x_offset, width, y_offset, height, image_depth, mask, fx_d, fy_d, cx_d, cy_d, depthScale, half_cube_dimension)
        
    #     if x != None:
    #         object_pose = PoseStamped()
    #         object_pose.header = depth_frame.header
    #         object_pose.pose.position.x = x
    #         object_pose.pose.position.y = y
    #         object_pose.pose.position.z = z
    #         object_pose.pose.orientation.x = 0
    #         object_pose.pose.orientation.y = 0
    #         object_pose.pose.orientation.z = 0
    #         object_pose.pose.orientation.w = 1
    #     else:
    #         return None

    #     return object_pose

    # def __trackCallback(self,data):
    #     self.tracked_objects = data

    # def get_objects_tracker(self, confidence):
    #     # get object message
    #     tracked_objects = rospy.wait_for_message("/bayes_objects_tracker/tracked_objects", TrackedObject3DList)
    #     object_frame = tracked_objects.header.frame_id
    #     tracked_objects = tracked_objects.objects

    #     object_dict = dict()
    #     for object in tracked_objects:
    #         obj_index = np.argmax(object.class_probability)

    #         if object.class_probability[obj_index] > confidence:
    #             new_object_name = object.class_name[obj_index]
    #             new_object_info = [object.uuid, object.class_probability[obj_index],
    #                             object.pose.pose.position.x, object.pose.pose.position.y, object.pose.pose.position.z,
    #                             object.pose.pose.orientation.x, object.pose.pose.orientation.x, object.pose.pose.orientation.x,
    #                             object.pose.pose.orientation.w]

    #             # check if object already exists and remove it
    #             repeated_obj_uuid, pruned_dict = self.__check_object_repetition(new_object_name, new_object_info, object_dict, obj_min_distance=0.2, remove_obj=True)
    #             if repeated_obj_uuid:
    #                 object_dict = pruned_dict

    #             # add new perceived object
    #             if new_object_name in object_dict.keys():
    #                 object_dict[new_object_name].append(new_object_info)
    #             else:
    #                 object_dict[new_object_name] = [new_object_info]

    #     return object_dict, object_frame

    # def __check_object_repetition(self, new_object_name, new_object_info, dict_objects, obj_min_distance=0.2, remove_obj=True):
    #     if remove_obj:
    #         pruned_dict = dict_objects.copy()
    #     new_obj_uuid = new_object_info[0]
    #     new_obj_pos = [new_object_info[2], new_object_info[3], new_object_info[4]]

    #     for obj_name, obj_info_list in dict_objects.items():
    #         for obj_index, obj_info in enumerate(obj_info_list):
    #             obj_uuid = obj_info[0]
    #             obj_pos = [obj_info[2], obj_info[3], obj_info[4]]
    #             if obj_uuid == new_obj_uuid or self.__euclidean_distance(new_obj_pos, obj_pos) < obj_min_distance:
    #                 if remove_obj:
    #                     del pruned_dict[obj_name][obj_index]
    #                     if not pruned_dict[obj_name]:
    #                         del pruned_dict[obj_name]

    #                     return obj_uuid, pruned_dict
    #                 else:
    #                     return obj_uuid, None

    #     return None, None

    # def __euclidean_distance(self, p1, p2):
    #     return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    # def start_tracker(self):
    # 	rospy.loginfo("Initializing object tracker...")
    # 	object_tracker_init = rospy.Publisher('/bayes_objects_tracker/event_in', String, queue_size=1)
    # 	rospy.sleep(0.5)
    # 	object_tracker_init.publish('e_start')

    # def stop_tracker(self):
    #     rospy.loginfo("Stopping object tracker...")
    #     object_tracker_init = rospy.Publisher('/bayes_objects_tracker/event_in', String, queue_size=1)
    #     rospy.sleep(0.5)
    #     object_tracker_init.publish('e_stop')

    # def delete_tracked_objects(self):
    #     self.start_tracker()
    #     rospy.sleep(0.1)
    #     tracked_objects = rospy.wait_for_message("/bayes_objects_tracker/tracked_objects", TrackedObject3DList)
    #     object_frame = tracked_objects.header.frame_id
    #     tracked_objects = tracked_objects.objects

    #     self.stop_tracker()
    #     clear_tracker_srv = rospy.ServiceProxy('/bayes_objects_tracker/delete_object', DeleteObject3D)
    #     rospy.sleep(0.1)
    #     clear_tracker_srv.wait_for_service(timeout=rospy.Duration(5))

    #     for obj in tracked_objects:
    #         delete_obj = DeleteObject3DRequest()
    #         delete_obj.uuid = obj.uuid
    #         rospy.loginfo(delete_obj)

    #         try:
    #             clear_tracker_srv.call(delete_obj)
    #         except:
    #             continue
    #         # rospy.sleep(0.1)

    #     self.start_tracker()

    # def get_number_of_objects(self, object_name, confidence=0.8):
    #     objects_dict = self.get_objects_tracker(confidence=confidence)
    #     if object_name in objects_dict.keys():
    #         return len(object_list[object_name])
    #     else:
    #         return 0

    # # TODO: implement get_objects_classes
    # def detect_object(self, class_to_detect):
    #     pass

    # # TODO: implement get_objects_classes
    # def get_objects_classes(self, perceive_time, confidence):
    #     pass

    # TODO: finish implementing get_objects_locations (check object repetition is not adapted to this message type, but the rest is done)
    # def get_objects_locations(self, perceive_time, confidence):
    #     tracked_objects = []
    #     initial_time = rospy.get_time()
    #     elapsed_time = -np.inf
    #     while perceive_time > elapsed_time:
    #         obj_msg = rospy.wait_for_message("/localized_objects", RecognizedObject3DList)
    #         object_frame = obj_msg.image_header.frame_id
    #         for obj in obj_msg.objects
    #             tracked_objects.append(obj)
    #         elapsed_time = rospy.get_time() - initial_time
    #
    #     object_dict = dict()
    #     for object in tracked_objects:
    #         if object.confidence > confidence:
    #             new_object_name = object.class_name
    #             new_object_info = [object.confidence,
    #                             object.pose.position.x, object.pose.position.y, object.pose.position.z,
    #                             object.pose.orientation.x, object.pose.pose.orientation.x, object.pose.orientation.x,
    #                             object.pose.orientation.w]
    #
    #             # check if object already exists and remove it
    #             repeated_obj_uuid, pruned_dict = self.__check_object_repetition(new_object_name, new_object_info, object_dict, obj_min_distance=0.2, remove_obj=True)
    #             if repeated_obj_uuid:
    #                 object_dict = pruned_dict
    #
    #             # add new perceived object
    #             if new_object_name in object_dict.keys():
    #                 object_dict[new_object_name].append(new_object_info)
    #             else:
    #                 object_dict[new_object_name] = [new_object_info]
    #
    #     return object_dict, object_frame

# Main function
if __name__ == '__main__':
    node_name = "perception_action"
    rospy.init_node(node_name, anonymous=True)
    rospy.loginfo("%s node created" % node_name)

    n_percep = Perception()
    # obj, depth = n_percep.detectPointingObjectWithCustomMsg(['suitcase', 'handbag', 'bag', 'backpack'])
    # rospy.loginfo(obj.object.class_name)
    # rospy.loginfo(obj.object.bounding_box)
    
    n_percep.downloadDetectronModel(3)
    rospy.loginfo("Done")