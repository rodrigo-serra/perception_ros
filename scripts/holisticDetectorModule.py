import cv2
import mediapipe as mp
import math

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
            return self.poseCoordinates[number].visibility >= 0.98
        return False


    def imgVisibilityCheck(self, number):
        if self.imgPoseCoordinates != []:
            return self.imgPoseCoordinates[number].visibility >= 0.98
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

    def getMiddlePoint(self, num1, num2):
        if self.visibilityCheck(num1) and self.visibilityCheck(num2):
            x = (self.poseCoordinates[num1].x + self.poseCoordinates[num2].x) / 2
            y = (self.poseCoordinates[num1].y + self.poseCoordinates[num2].y) / 2
            z = (self.poseCoordinates[num1].z + self.poseCoordinates[num2].z) / 2
            return [x, y, z]
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
        rightArmLength = self.getArmLenght(self.mpHolistic.PoseLandmark.RIGHT_SHOULDER, 
                                            self.mpHolistic.PoseLandmark.RIGHT_ELBOW, 
                                            self.mpHolistic.PoseLandmark.RIGHT_WRIST)
        if rightArmLength != -1:
            return rightArmLength
        else:
            return False


    def getLeftArmLength(self):
        leftArmLength = self.getArmLenght(self.mpHolistic.PoseLandmark.LEFT_SHOULDER, 
                                            self.mpHolistic.PoseLandmark.LEFT_ELBOW, 
                                            self.mpHolistic.PoseLandmark.LEFT_WRIST)
        if leftArmLength != -1:
            return leftArmLength
        else:
            return False


    def getShoulderLength(self):
        shoulderLength = self.distanceBetweenPoints(self.mpHolistic.PoseLandmark.LEFT_SHOULDER, 
                                                    self.mpHolistic.PoseLandmark.RIGHT_SHOULDER)
        if shoulderLength != -1:
            return shoulderLength
        else:
            return False
    

    def getHipLength(self):
        hipLength = self.distanceBetweenPoints(self.mpHolistic.PoseLandmark.LEFT_HIP, 
                                                self.mpHolistic.PoseLandmark.RIGHT_HIP)
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


    def getPointingDirectionRightHand(self, img, drawPoitingDirectionSlope = False):
        if self.rightHandCoordinates != []:
            x1 = self.rightHandCoordinates[self.mpHolistic.HandLandmark.WRIST][0]
            y1 = self.rightHandCoordinates[self.mpHolistic.HandLandmark.WRIST][1]

            x2 = self.rightHandCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_TIP][0]
            y2 = self.rightHandCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_TIP][1]

            m, b, px, py, qx, qy = self.slopePointingDirection(img, x1, y1, x2, y2)

            if drawPoitingDirectionSlope:
                cv2.line(img, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 2)

        return img, m, b

    
    def getPointingDirectionLeftHand(self, img, drawPoitingDirectionSlope = False):
        if self.leftHandCoordinates != []:
            x1 = self.leftHandCoordinates[self.mpHolistic.HandLandmark.WRIST][0]
            y1 = self.leftHandCoordinates[self.mpHolistic.HandLandmark.WRIST][1]

            x2 = self.leftHandCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_TIP][0]
            y2 = self.leftHandCoordinates[self.mpHolistic.HandLandmark.INDEX_FINGER_TIP][1]

            m, b, px, py, qx, qy = self.slopePointingDirection(img, x1, y1, x2, y2)

            if drawPoitingDirectionSlope:
                cv2.line(img, (int(px), int(py)), (int(qx), int(qy)), (255, 255, 0), 2)

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