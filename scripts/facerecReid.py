#!/usr/bin/env python3

import rospy
import rospkg
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from PIL import Image as imgPil
from facerecModule import *



class Reid:
    def __init__(self):
        rospy.init_node('reid')
        rospack = rospkg.RosPack()

        # Variable Initialization
        self.directory = rospack.get_path('perception_tests') + '/images/'
        self.rate = rospy.Rate(10)
        self.img = None
        self.draw = True
        self.personCounter = 0
        self.ctr = True

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
                if self.ctr:
                    face_locations, face_names = faceRecognition(self.img, self.known_face_encodings, self.known_face_names)
                    
                    if face_names != []:
                        res = self.lookIntoDetectPeople(face_locations, face_names)
                        rospy.loginfo(res)
                        self.reid_pub.publish(res)
                    else:
                        rospy.loginfo("No detections!")
                        self.reid_pub.publish("No detections!")

                    self.ctr = False
                    
                if self.draw:
                    frame = drawRectangleAroundFace(self.img, face_locations, face_names)
                    cv2.imshow("RealSense", frame)
                    cv2.waitKey(1)
                    
            
            self.rate.sleep()

        if self.draw:
            cv2.destroyAllWindows()
        rospy.loginfo('Shutting Down Reid Node')


    def lookIntoDetectPeople(self, face_locations, face_names):
        detectionResult = ''
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if name == "Unknown":
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
                    self.known_face_names.append("H #" + str(self.personCounter))
                    self.personCounter += 1
            
            
            detectionResult += name + '; '
        
        return detectionResult

    def imgCallback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.ctr = True
        except CvBridgeError as e:
            print(e)



# Main function
if __name__ == '__main__':
    camera = Reid()
    camera.run()