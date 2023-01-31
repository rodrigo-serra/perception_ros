#!/usr/bin/env python3

import rospy
import rospkg
import cv2
import numpy as np
import os

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from perception_tests.msg import StringArray
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as imgPil
from facerecModule import *
from holisticDetectorModule import *
from PIL import ImageDraw
from scipy.spatial import ConvexHull



class Reid:
    def __init__(self):
        # Create the node
        node_name = "reid"
        rospy.init_node(node_name, anonymous=False)
        rospy.loginfo("%s node created" % node_name)

        rospack = rospkg.RosPack()

        # Variable Initialization
        self.directory = rospack.get_path('perception_tests') + '/images/'
        self.rate = rospy.Rate(10)
        self.img = None
        self.personCounter = 0
        self.ctr = True
        self.detector = holisticDetector()
        self.cropOffset = 50
        self.currentEvent = None
        self.takePhoto = False
        self.runAutomatic = False
        self.bridge = CvBridge()
        

        # Create arrays of known face encodings and their names
        self.known_face_encodings = []
        self.known_face_names = []
        

        # Read from ROS Param
        self.camera_topic = rospy.get_param("~camera_topic")
        self.readImgCompressed = rospy.get_param("~img_compressed")
        self.draw = rospy.get_param("~visualization")
        self.extractFaceBoundaryOnly = rospy.get_param("~extract_face_boundary_only")

        # Subscribe to Camera Topic
        if self.readImgCompressed:
            self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
        else:
            self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)

        # Subscribe to Event and perform accordingly (start, stop, restart, automatic or non-automatic modes, take photo)
        self.event_sub = rospy.Subscriber("~event_in", String, self.eventCallback)

        # Publish Detected Faces
        self.reid_pub = rospy.Publisher("~current_detection", StringArray, queue_size=10)

        # Publish Record of Detected Faces
        self.reidRecord_pub = rospy.Publisher("~detection_record", StringArray, queue_size=10)

    def run(self):
        while not rospy.is_shutdown():
            if self.currentEvent is not None:
                if self.currentEvent == "take_photo":
                    self.takePhoto = True
                    self.currentEvent = None
                    rospy.loginfo("Taking photo to unknow person!")

                if self.currentEvent == "enable_automatic":
                    self.runAutomatic = True
                    self.currentEvent = None
                    rospy.loginfo("Enabling automatic mode!")

                if self.currentEvent == "disable_automatic":
                    self.runAutomatic = False
                    self.currentEvent = None
                    rospy.loginfo("Disabling automatic mode!")

                if self.currentEvent == "stop":
                    self.currentEvent = None
                    self.img = None
                    self.image_sub.unregister()
                    cv2.destroyAllWindows()
                    rospy.loginfo("Stopping detection!")

                if self.currentEvent == "start":
                    self.currentEvent = None
                    if self.readImgCompressed:
                        self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
                    else:
                        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)
                    rospy.loginfo("Starting detection!")

                if self.currentEvent == "reset":
                    self.deleteAllImgs()
                    self.img = None
                    self.personCounter = 0
                    self.ctr = True
                    self.detector = holisticDetector()
                    self.cropOffset = 50
                    self.currentEvent = None
                    self.takePhoto = False
                    self.runAutomatic = False

                    if self.readImgCompressed:
                        self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.imgCallback)
                    else:
                        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.imgCallback)

                    # Create arrays of known face encodings and their names
                    self.known_face_encodings = []
                    self.known_face_names = []
                    
                    rospy.loginfo("Reseting!")

            if self.img is not None:
                if self.ctr:
                    face_locations, face_names = faceRecognition(self.img, self.known_face_encodings, self.known_face_names)
                    
                    if face_names != []:
                        if self.extractFaceBoundaryOnly:
                            res = self.lookIntoDetectPeopleHolistic(face_locations, face_names)
                        else:
                            res = self.lookIntoDetectPeople(face_locations, face_names)

                        # rospy.loginfo(res)
                        self.reid_pub.publish(res)
                    else:
                        rospy.loginfo("No detections!")

                    if self.known_face_names != []:
                        self.reidRecord_pub.publish(self.known_face_names)
                    
                    self.ctr = False
                    
                if self.draw:
                    frame = drawRectangleAroundFace(self.img, face_locations, face_names)
                    cv2.imshow("RealSense", frame)
                    cv2.waitKey(1)
                    
            
            self.rate.sleep()

        if self.draw:
            cv2.destroyAllWindows()
        rospy.loginfo('Shutting Down Reid Node')


    def getFaceMask(self, img):
        height, width, c = img.shape
        mask_img = imgPil.new('L', (width, height), 0)

        points = np.array(self.detector.faceCoordinates)

        hull = ConvexHull(points)
        polygon = []
        for v in hull.vertices:
            polygon.append((points[v, 0], points[v, 1]))

        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
        mask = np.array(mask_img)
        
        color_image = cv2.bitwise_and(img, img, mask=mask)

        return color_image


    def lookIntoDetectPeopleHolistic(self, face_locations, face_names):
        detectionResult = []
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if (name == "Unknown" and self.takePhoto) or (name == "Unknown" and self.runAutomatic):
                # Draw Mask
                mask = np.zeros(self.img.shape[:2], dtype="uint8")
                left -= self.cropOffset
                top -= self.cropOffset
                right += self.cropOffset
                bottom += self.cropOffset
                cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
                img_masked = cv2.bitwise_and(self.img, self.img, mask=mask)

                img_masked = self.detector.find(img_masked, False, False, False, False)
                isFaceLandmarks = self.detector.getFaceLandmarks(img_masked)

                if isFaceLandmarks:
                    img_masked = self.getFaceMask(img_masked)
                    # Save Img and Add to Enconder
                    imageRGB = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
                    # Save new image of Person (not required)
                    im = imgPil.fromarray(imageRGB)
                    im_name = self.directory + "h" + str(self.personCounter) + ".png"
                    im.save(im_name)
                    # Load new Person to enconder
                    faceEnconder = face_recognition.face_encodings(imageRGB)
                    if len(faceEnconder) > 0:
                        self.known_face_encodings.append(faceEnconder[0])
                        self.known_face_names.append("H" + str(self.personCounter))
                        self.personCounter += 1
                        self.takePhoto = False
                        rospy.loginfo("Photo saved and added to enconder!")
           
            
            detectionResult.append(name)
        
        return detectionResult


    def lookIntoDetectPeople(self, face_locations, face_names):
        detectionResult = []
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if (name == "Unknown" and self.takePhoto) or (name == "Unknown" and self.runAutomatic):
                rospy.loginfo("Taking photo to unknow person!")
                # Draw Mask
                mask = np.zeros(self.img.shape[:2], dtype="uint8")
                cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
                # cv2.imshow("Rectangular Mask", mask)
                img_masked = cv2.bitwise_and(self.img, self.img, mask=mask)
                # Save Img and Add to Enconder
                imageRGB = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
                # Save new image of Person (not required)
                im = imgPil.fromarray(imageRGB)
                im_name = self.directory + "h" + str(self.personCounter) + ".png"
                im.save(im_name)
                # Load new Person to enconder
                faceEnconder = face_recognition.face_encodings(imageRGB)
                if len(faceEnconder) > 0:
                    self.known_face_encodings.append(faceEnconder[0])
                    self.known_face_names.append("H" + str(self.personCounter))
                    self.personCounter += 1
                    self.takePhoto = False
                    rospy.loginfo("Photo saved and added to enconder!")
            
            
            detectionResult.append(name)
        
        return detectionResult



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
        # rospy.loginfo("Got new event: " + str(data.data))
        self.currentEvent = data.data


    def deleteAllImgs(self):
        for file_name in os.listdir(self.directory):
            file = self.directory + file_name
            if os.path.isfile(file):
                os.remove(file)



# Main function
if __name__ == '__main__':
    camera = Reid()
    camera.run()