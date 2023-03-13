import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
from PIL import Image as IMG
from PIL import ImageEnhance


class tridimensionalInfo():
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class bidimensionalInfo():
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility


class holisticDetector():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic()
        self.visibilityThreshold = 0.9
        self.handDistanceToBodyThreshold = 0.3
        self.rightHandReturnMsg = "Right Hand"
        self.leftHandReturnMsg = "Left Hand"


    def find(self, img, pose, face, rightHand, leftHand):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
    
        if pose and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)

        if face and self.results.face_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_TESSELATION)
            # self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_CONTOURS)

        if rightHand and self.results.right_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)

        if leftHand and self.results.left_hand_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)

        return img


    def getFaceLandmarks(self, img):
        self.faceCoordinates = []
        h, w, c = img.shape
        if self.results.face_landmarks:
            for id, lm in enumerate(self.results.face_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.faceCoordinates.append((cx, cy))
            return True
        return False


    def getPoseWorldLandmarks(self):
        self.poseCoordinates = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                self.poseCoordinates.append(tridimensionalInfo(lm.x, lm.y, lm.z, lm.visibility))
            return True
        return False


    def getPoseImgLandmarks(self, img):
        self.imgPoseCoordinates = []
        h, w, c = img.shape
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.imgPoseCoordinates.append(bidimensionalInfo(cx, cy, lm.visibility))
            return True
        return False


    def getRightHandLandmarks(self, img):
        self.rightHandCoordinates = []
        h, w, c = img.shape
        if self.results.right_hand_landmarks:
            for id, lm in enumerate(self.results.right_hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.rightHandCoordinates.append((cx, cy))
            return True
        return False


    def getLeftHandLandmarks(self, img):
        self.leftHandCoordinates = []
        h, w, c = img.shape
        if self.results.left_hand_landmarks:
            for id, lm in enumerate(self.results.left_hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.leftHandCoordinates.append((cx, cy))
            return True
        return False


    def visibilityCheck(self, number):
        if self.poseCoordinates != []:
            return self.poseCoordinates[number].visibility >= self.visibilityThreshold
        return False


    def imgVisibilityCheck(self, number):
        if self.imgPoseCoordinates != []:
            return self.imgPoseCoordinates[number].visibility >= self.visibilityThreshold
        return False


    def printPointCoordinates(self, number):
        if self.visibilityCheck(number):
            print("ID :" + str(number))
            print("x: " + str(self.poseCoordinates[number].x))
            print("y: " + str(self.poseCoordinates[number].y))
            print("z: " + str(self.poseCoordinates[number].z))
        else:
            print("Visibility of point " + str(number) + " is too low")


    def printImgPointCoordinates(self, number):
        if self.imgVisibilityCheck(number):
            print("ID :" + str(number))
            print("u: " + str(self.imgPoseCoordinates[number].x))
            print("v: " + str(self.imgPoseCoordinates[number].y))
        else:
            print("Visibility of point " + str(number) + " is too low")


    def returnImgPointCoordinates(self, number):
        if self.imgVisibilityCheck(number):
            return self.imgPoseCoordinates[number].x, self.imgPoseCoordinates[number].y
        else:
            print("Visibility of point " + str(number) + " is too low")
            return -1, -1


    def distanceBetweenPoints(self, num1, num2):
        if self.visibilityCheck(num1) and self.visibilityCheck(num2):
            return self.distanceFormula(self.poseCoordinates[num1].x, 
                                        self.poseCoordinates[num1].y, 
                                        self.poseCoordinates[num1].z, 
                                        self.poseCoordinates[num2].x, 
                                        self.poseCoordinates[num2].y, 
                                        self.poseCoordinates[num2].z)
        else:
            return -1

    def distanceFormula(self, x1, y1, z1, x2, y2, z2):
        dx = math.pow(x1 - x2, 2)
        dy = math.pow(y1- y2, 2)
        dz = math.pow(z1 - z2, 2)
        return math.sqrt(dx + dy + dz)
    

    def distanceFormula2D(self, x1, y1, x2, y2):
        dx = math.pow(x1 - x2, 2)
        dy = math.pow(y1- y2, 2)
        return math.sqrt(dx + dy)


    def getMiddlePoint(self, num1, num2):
        if self.visibilityCheck(num1) and self.visibilityCheck(num2):
            x = (self.poseCoordinates[num1].x + self.poseCoordinates[num2].x) / 2
            y = (self.poseCoordinates[num1].y + self.poseCoordinates[num2].y) / 2
            z = (self.poseCoordinates[num1].z + self.poseCoordinates[num2].z) / 2
            return [x, y, z]
        else:
            return -1
    
    
    def getMiddlePointImg(self, num1, num2):
        if self.imgVisibilityCheck(num1) and self.imgVisibilityCheck(num2):
            x = int((self.imgPoseCoordinates[num1].x + self.imgPoseCoordinates[num2].x) / 2)
            y = int((self.imgPoseCoordinates[num1].y + self.imgPoseCoordinates[num2].y) / 2)
            return [x, y]
        else:
            return -1

    
    def getArmLenght(self, num1, num2, num3):
        arm_1 = self.distanceBetweenPoints(num1, num2)
        arm_2 = self.distanceBetweenPoints(num2, num3)

        if arm_1 != -1 and arm_2!= -1:
            return arm_1 + arm_2
        else:
            return -1

    
    def getRightArmLength(self):
        rightArmLength = self.getArmLenght(self.mpHolistic.PoseLandmark.RIGHT_SHOULDER, self.mpHolistic.PoseLandmark.RIGHT_ELBOW, self.mpHolistic.PoseLandmark.RIGHT_WRIST)
        if rightArmLength != -1:
            return rightArmLength
        else:
            return False


    def getLeftArmLength(self):
        leftArmLength = self.getArmLenght(self.mpHolistic.PoseLandmark.LEFT_SHOULDER, self.mpHolistic.PoseLandmark.LEFT_ELBOW, self.mpHolistic.PoseLandmark.LEFT_WRIST)
        if leftArmLength != -1:
            return leftArmLength
        else:
            return False


    def getShoulderLength(self):
        shoulderLength = self.distanceBetweenPoints(self.mpHolistic.PoseLandmark.LEFT_SHOULDER, self.mpHolistic.PoseLandmark.RIGHT_SHOULDER)
        if shoulderLength != -1:
            return shoulderLength
        else:
            return False
    

    def getHipLength(self):
        hipLength = self.distanceBetweenPoints(self.mpHolistic.PoseLandmark.LEFT_HIP, self.mpHolistic.PoseLandmark.RIGHT_HIP)
        if hipLength != -1:
            return hipLength
        else:
            return False

    
    def getTorsoLength(self):
        middlePoint_1 = self.getMiddlePoint(self.mpHolistic.PoseLandmark.LEFT_SHOULDER, self.mpHolistic.PoseLandmark.RIGHT_SHOULDER)
        middlePoint_2 = self.getMiddlePoint(self.mpHolistic.PoseLandmark.LEFT_HIP, self.mpHolistic.PoseLandmark.RIGHT_HIP)

        if middlePoint_1 != -1 and middlePoint_2 != -1:
            torsoLen = self.distanceFormula(
                middlePoint_1[0],
                middlePoint_1[1],
                middlePoint_1[2],
                middlePoint_2[0],
                middlePoint_2[1],
                middlePoint_2[2],
            )
            return torsoLen
        else:
            return False


    def getPointingArm(self):
        rightHandDistanceToBody = self.distanceBetweenPoints(self.mpHolistic.PoseLandmark.RIGHT_HIP, self.mpHolistic.PoseLandmark.RIGHT_WRIST)
        leftHandDistanceToBody = self.distanceBetweenPoints(self.mpHolistic.PoseLandmark.LEFT_HIP,self.mpHolistic.PoseLandmark.LEFT_WRIST)

        if rightHandDistanceToBody == -1 and leftHandDistanceToBody == -1:
            return False

        if rightHandDistanceToBody > leftHandDistanceToBody and rightHandDistanceToBody > self.handDistanceToBodyThreshold:
            # print("Right Hand Distance: " + str(rightHandDistanceToBody))
            return self.rightHandReturnMsg

        if rightHandDistanceToBody < leftHandDistanceToBody and leftHandDistanceToBody > self.handDistanceToBodyThreshold:
            # print("Left Hand Distance: " + str(leftHandDistanceToBody))
            return self.leftHandReturnMsg
        
        return False

    
    def getPointingDirectionArm(self, img, whichHand, drawPoitingDirectionSlope = True):
        m, b = None, None
        if whichHand == self.rightHandReturnMsg and self.imgPoseCoordinates != []:
            x1 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.RIGHT_ELBOW].x
            y1 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.RIGHT_ELBOW].y
            x2 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.RIGHT_WRIST].x
            y2 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.RIGHT_WRIST].y
        elif whichHand == self.leftHandReturnMsg and self.imgPoseCoordinates != []:
            x1 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.LEFT_ELBOW].x
            y1 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.LEFT_ELBOW].y
            x2 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.LEFT_WRIST].x
            y2 = self.imgPoseCoordinates[self.mpHolistic.PoseLandmark.LEFT_WRIST].y
        else:
            return img, m, b

        m, b, px, py, qx, qy = self.slopePointingDirection(img, x1, y1, x2, y2)

        if drawPoitingDirectionSlope:
                cv2.line(img, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 2)

        return img, m, b


    def getPointingDirectionHand(self, img, whichHand, drawPoitingDirectionSlope = True):
        handCoordinates = []
        m, b = None, None

        if whichHand == self.rightHandReturnMsg and self.rightHandCoordinates != []:
            handCoordinates = self.rightHandCoordinates
        elif whichHand == self.leftHandReturnMsg and self.leftHandCoordinates != []:
            handCoordinates = self.leftHandCoordinates
        else:
            return img, m, b

        # x1 = handCoordinates[self.mpHolistic.HandLandmark.WRIST][0]
        # y1 = handCoordinates[self.mpHolistic.HandLandmark.WRIST][1]

        x1 = handCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_MCP][0]
        y1 = handCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_MCP][1]

        # x1 = handCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_PIP][0]
        # y1 = handCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_PIP][1]

        x2 = handCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_TIP][0]
        y2 = handCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_TIP][1]

        m, b, px, py, qx, qy = self.slopePointingDirection(img, x1, y1, x2, y2)

        if drawPoitingDirectionSlope:
                cv2.line(img, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 2)

        return img, m, b


    def slopePointingDirection(self, img, x1, y1, x2, y2):
        h, w, c = img.shape
        
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            px, qx = 0, w
            py, qy = m * px + b, m * qx + b
        else:
            m = None
            b = None
            px, py = x1, 0
            qx, qy = x1, h


        return m, b, px, py, qx, qy

    def readSweaterColor(self, img, pkg_path):
        mPoint = self.getMiddlePointImg(self.mpHolistic.PoseLandmark.LEFT_SHOULDER, self.mpHolistic.PoseLandmark.RIGHT_SHOULDER)
        if mPoint == -1:
            return False

        # Select a point a little bit below mPoint
        mPoint[1] += 50
        
        # Change contrast and brightness
        applyContrast = False
        if applyContrast:
            # Contrast control
            alpha = 1.5
            # Brightness control
            beta = 10
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


        # Change saturation
        applySaturation = True
        if applySaturation:
            color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = IMG.fromarray(color_coverted)
            img = ImageEnhance.Color(pil_image)
            img = img.enhance(4.0)
            img = np.array(img)
            img = img[:, :, ::-1].copy()  #Convert RGB to BGR 


        # Select region of interest
        offset = 20
        cx_left = mPoint[0] - offset
        cx_right = mPoint[0] + offset

        cy_top = mPoint[1] - offset
        cy_bottom = mPoint[1] + offset

        new_img = img[cx_left:cx_right, cy_top:cy_bottom, :]

        img_blue_channel = img[cx_left:cx_right, cy_top:cy_bottom, 0]
        img_green_channel = img[cx_left:cx_right, cy_top:cy_bottom, 1]
        img_red_channel = img[cx_left:cx_right, cy_top:cy_bottom, 2]

        # Apply Median or Average on region of interest
        apply_median = True
        
        if np.all(img_blue_channel != img_blue_channel):
            return False
        blue = np.average(img_blue_channel)
        blue = int(blue)

        if np.all(img_green_channel != img_green_channel):
            return False
        green = np.average(img_green_channel)
        green = int(green)
        
        if np.all(img_red_channel != img_red_channel):
            return False
        red = np.average(img_red_channel)
        red = int(red)

        if apply_median:
            blue = int(np.median(img_blue_channel))
            green = int(np.median(img_green_channel))
            red = int(np.median(img_red_channel))

        # Get Color
        color_name = self.getColorName(pkg_path, red, green, blue)
        return color_name

    
    def getColorName(self, pkg_path, R, G, B):
        # Read CSV with color codes
        file_name = pkg_path + "/files/basic_colors_simplified.csv"
        index=["color","color_name","hex","R","G","B"]
        csv = pd.read_csv(file_name, names=index, header=None)
    
        minimum = 10000
        cname = ""
        for i in range(len(csv)):
            d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
            if(d<=minimum):
                minimum = d
                cname = csv.loc[i,"color_name"]
        return cname