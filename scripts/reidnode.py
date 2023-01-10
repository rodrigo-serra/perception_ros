#!/usr/bin/env python3

import rospy
import rospkg
import cv2
import numpy as np
import os

from std_msgs.msg import String
from sensor_msgs.msg import Image
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
        self.draw = True
        self.personCounter = 0
        self.ctr = True
        self.detector = holisticDetector()
        self.cropOffset = 50
        
        self.currentEvent = None
        self.runHolistic = True
        self.takePhoto = False
        self.runAutomatic = False

        # Create arrays of known face encodings and their names
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.bridge = CvBridge()

        # Subscribe to Camera Topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.imgCallback)

        # Subscribe to Event and perform accordingly (start, stop, restart, automatic or non-automatic modes, take photo)
        self.event_sub = rospy.Subscriber("/reid/event_in", String, self.eventCallback)

        # Publish Detected Faces
        self.reid_pub = rospy.Publisher("/reid/results", String, queue_size=10)

    def run(self):
        while not rospy.is_shutdown():
            if self.currentEvent is not None:
                if self.currentEvent == "take_photo":
                    self.takePhoto = True
                    self.currentEvent = None
                    rospy.logwarn("Taking photo to unknow person!")

                if self.currentEvent == "enable_automatic":
                    self.runAutomatic = True
                    self.currentEvent = None
                    rospy.logwarn("Enabling automatic mode!")

                if self.currentEvent == "disable_automatic":
                    self.runAutomatic = False
                    self.currentEvent = None
                    rospy.logwarn("Disabling automatic mode!")

                if self.currentEvent == "stop":
                    self.currentEvent = None
                    self.img = None
                    self.image_sub.unregister()
                    cv2.destroyAllWindows()
                    rospy.logwarn("Stopping detection!")

                if self.currentEvent == "start":
                    self.currentEvent = None
                    self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.imgCallback)
                    rospy.logwarn("Starting detection!")

                if self.currentEvent == "reset":
                    self.deleteAllImgs()
                    self.img = None
                    self.draw = True
                    self.personCounter = 0
                    self.ctr = True
                    self.detector = holisticDetector()
                    self.cropOffset = 50
                    
                    self.currentEvent = None
                    self.runHolistic = True
                    self.takePhoto = False
                    self.runAutomatic = False

                    self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.imgCallback)

                    # Create arrays of known face encodings and their names
                    self.known_face_encodings = []
                    self.known_face_names = []
                    
                    rospy.logwarn("Reseting!")

            if self.img is not None:
                if self.ctr:
                    face_locations, face_names = faceRecognition(self.img, self.known_face_encodings, self.known_face_names)
                    
                    if face_names != []:
                        if self.runHolistic:
                            res = self.lookIntoDetectPeopleHolistic(face_locations, face_names)
                        else:
                            res = self.lookIntoDetectPeople(face_locations, face_names)

                        rospy.loginfo(res)
                        self.reid_pub.publish(res)
                    else:
                        rospy.loginfo("No detections!")
                        self.reid_pub.publish("No detections")

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
        detectionResult = ''
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
                        rospy.logwarn("Photo saved and added to enconder!")
           
            
            detectionResult += name + ';'
        
        return detectionResult


    def lookIntoDetectPeople(self, face_locations, face_names):
        detectionResult = ''
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if (name == "Unknown" and self.takePhoto) or (name == "Unknown" and self.runAutomatic):
                rospy.logwarn("Taking photo to unknow person!")
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
                    rospy.logwarn("Photo saved and added to enconder!")
            
            
            detectionResult += name + ';'
        
        return detectionResult



    def imgCallback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.ctr = True
        except CvBridgeError as e:
            print(e)


    def eventCallback(self, data):
        # rospy.logwarn("Got new event: " + str(data.data))
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