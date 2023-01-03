#!/usr/bin/env python3

import rospy
import rospkg
import cv2
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from holisticDetectorModule import *
from PIL import Image as imgPil
from PIL import ImageDraw
from scipy.spatial import ConvexHull

class Holistic:
    def __init__(self):
        rospy.init_node('holistic')

        # Variable Initialization
        self.rate = rospy.Rate(10)
        self.img = None
        self.ctr = True
        self.detector = holisticDetector()

        self.bridge = CvBridge()

        # Subscribe to Camera Topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.imgCallback)

        # Publish Detected Faces
        self.holistic_pub = rospy.Publisher("/reid", String, queue_size=10)

    def run(self):
        while not rospy.is_shutdown():
            if self.img is not None:
                if self.ctr:
                    self.img = self.detector.find(self.img, False, False, False, False)
                    isFaceLandmarks = self.detector.getFaceLandmarks(self.img)
                    self.holistic_pub.publish(str(isFaceLandmarks))
                    self.ctr = False
                    if isFaceLandmarks:
                        self.img = self.getFaceMask()
                
                cv2.imshow("RealSense", self.img)
                cv2.waitKey(1)
            
            self.rate.sleep()

        
        cv2.destroyAllWindows()
        rospy.loginfo('Shutting Down Reid Node')


    def imgCallback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.ctr = True
        except CvBridgeError as e:
            print(e)


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



# Main function
if __name__ == '__main__':
    camera = Holistic()
    camera.run()