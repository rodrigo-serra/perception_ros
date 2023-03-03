#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_py.msg import RecognizedObjectArrayStamped
from perception_tests.msg import MediapipePointInfo, MediapipePointInfoArray
from perception_tests.msg import ReidInfoArray
from sympy import Point, Polygon, Line

# from mbot_perception_msgs.msg import TrackedObject3DList, TrackedObject3D, RecognizedObject3DList, RecognizedObject3D
# from mbot_perception_msgs.srv import DeleteObject3D, DeleteObject3DRequest

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Perception():
    __metaclass__ = Singleton
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
     

    def detectPointingObject(self, useYolo = True, easyDetection = False, useFilteredObjects = True, classNameToBeDetected = 'backpack', score = 0.5):
        self.__detectedObjects = self.returnDetectedObjects()
        if easyDetection:
            self.getPointingDirection()
            return self.__findObjectSimplifiedVersion()
        else:
            self.__getImg(useYolo)
            self.getPointingSlope()
            self.getPointingIntercept()
            
            res = self.__lineIntersectionPolygon()
            if res != None:
                return res
            else:
                return self.__findClosestObjectToLine()


    def returnDetectedObjects(self, useYolo = True, useFilteredObjects = True, classNameToBeDetected = 'backpack', score = 0.5):
        dObjects = []
        detectedObjects_topic = "/detectron2_ros/result_yolo_msg"
        if useYolo == True:
            detectedObjects_topic = "/object_detector/detections"

        try:
            data = rospy.wait_for_message(detectedObjects_topic, RecognizedObjectArrayStamped, timeout = self.__timeout)
        except:
            rospy.logerr("Object Detection Results are not being published!")
            exit(1)

        if useFilteredObjects:
            for obj in data.objects.objects:
                if obj.class_name == classNameToBeDetected and obj.confidence > score:
                    dObjects.append(obj)
        else:
            for obj in data.objects.objects:    
                dObjects.append(obj)
        
        return dObjects


    def __getImg(self, useYolo):
        bridge = CvBridge()
        camera_topic = "/camera/color/image_raw"
        readImgMsg = Image
        if useYolo == True:
            camera_topic = "/object_detector/detection_image/compressed"
            readImgMsg = CompressedImage

        try:
            data = rospy.wait_for_message(camera_topic, readImgMsg, timeout = self.__timeout)
        except:
            rospy.logerr("Could not read img from: " + camera_topic)

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


# if __name__ == '__main__':
#     rospy.init_node('perception_action', anonymous=True)
#     perception = Perception()
#     objects, object_frame = perception.get_objects_tracker(confidence=0.5)
#     print(objects)
#     print(object_frame)


def main():
    node_name = "perception_action"
    rospy.init_node(node_name, anonymous=True)
    rospy.loginfo("%s node created" % node_name)

    n_percep = Perception()
    
    obj = n_percep.getPeopleDetectionRecord()
    rospy.loginfo(obj)
    return obj

    # rospy.loginfo(n_percep.readSweaterColor())

# Main function
if __name__ == '__main__':
    main()