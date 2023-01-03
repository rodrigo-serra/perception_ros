#!/usr/bin/env python3

import rospy
import cv2
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from facerecModule import *



class Reid:
    def __init__(self):
        rospy.init_node('reid')
        
        # Variable Initialization
        self.rate = rospy.Rate(10)
        self.img = None
        self.draw = True

        # Create arrays of known face encodings and their names
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.bridge = CvBridge()

        # Subscribe to Camera Topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.imgCallback)

        # Publish Detected Faces
        self.reid_pub = rospy.Publisher("/reid", String, queue_size=10)

    def run(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                face_locations, face_names = faceRecognition(self.img, self.known_face_encodings, self.known_face_names)
                
                if face_names != []:
                    rospy.loginfo(face_names[0])
                    self.reid_pub.publish(face_names[0])
                else:
                    rospy.loginfo("No detections!")
                    self.reid_pub.publish("No detections!")

                if self.draw:
                    frame = drawRectangleAroundFace(self.img, face_locations, face_names)
                    cv2.imshow("RealSense", frame)
                    cv2.waitKey(1)
            
            self.rate.sleep()

        if self.draw:
            cv2.destroyAllWindows()
        rospy.loginfo('Shutting Down Reid Node')

                

    def imgCallback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)



# Main function
if __name__ == '__main__':
    camera = Reid()
    camera.run()