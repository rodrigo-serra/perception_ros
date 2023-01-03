import cv2
import mediapipe as mp
import math

class signature():
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors


class person():
    def __init__(self, id, keypoints, descriptors):
        self.id = id
        self.face_sign = signature(keypoints, descriptors)
        self.seen = 10


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


    def getPoseImgLandmarks(self, img):
        self.imgPoseCoordinates = []
        h, w, c = img.shape
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.imgPoseCoordinates.append(bidimensionalInfo(cx, cy, lm.visibility))
                # cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return img
        

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

