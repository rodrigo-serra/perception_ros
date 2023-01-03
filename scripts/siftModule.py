import cv2
import numpy as np


# Initialize SIFT
sift = cv2.SIFT_create()

# Feature Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)


def applySift(color_image, applyCrop, ux, uy, r):
    if applyCrop:
        # Crop Image
        color_image = color_image[uy - r:uy + r, ux - r:ux + r, :]  
    # Convert image to grayscale for SIFT
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # Extract SIFT features
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return color_image, gray_image, keypoints, descriptors


def drawKeypointsImg(gray_image, keypoints, color_image):
    # Draw Keypoints in image (single image)
    color_image = cv2.drawKeypoints(gray_image, keypoints, color_image)
    return color_image



def computeDistanceBetweenKeypoints(matches, prev_keypoints, keypoints):
    # Featured matched keypoints from images 1 and 2
    pts1 = np.float32([keypoints[m.queryIdx].pt for m in matches])
    pts2 = np.float32([prev_keypoints[m.trainIdx].pt for m in matches])

    # Convert x, y coordinates into complex numbers
    # so that the distances are much easier to compute
    z1 = np.array([[complex(c[0],c[1]) for c in pts1]])
    z2 = np.array([[complex(c[0],c[1]) for c in pts2]])

    # Computes the intradistances between keypoints for each image
    KP_dist1 = abs(z1.T - z1)
    KP_dist2 = abs(z2.T - z2)

    # Distance between featured matched keypoints
    FM_dist = abs(z2 - z1)
    print("Num of Matches: ")
    print(FM_dist.shape[1])
    print("Avg Distance Between Features (px): ")
    print(np.sum(FM_dist) / FM_dist.shape[1])


def applyMatching(descriptors, prev_descriptors, keypoints, prev_keypoints):
    # Match descriptors
    matches = bf.match(descriptors, prev_descriptors)
    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)

    return matches


def siftKeypointsMatching(ux, uy, color_image, prev_descriptors, prev_keypoints, i, drawKeypoints, cropImg, radius):
    color_image, gray_image, keypoints, descriptors = applySift(color_image, cropImg, ux, uy, radius)
    # Find Matches Between Current Frame and Previous Frame
    # Showed Them Live
    matches = []
    if i != 0:
        matches = applyMatching(descriptors, prev_descriptors, keypoints, prev_keypoints)
        
        # Get distance between keypoints in px
        # if len(matches) > 0:
        #     computeDistanceBetweenKeypoints(matches, prev_keypoints, keypoints)
        
        # Draw Keypoints in image (previous and current frame keypoints)
        if drawKeypoints:
            color_image = cv2.drawMatches(gray_image, 
                                        keypoints, 
                                        gray_image, 
                                        prev_keypoints, 
                                        matches[:len(matches)], 
                                        None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    prev_descriptors = descriptors
    prev_keypoints = keypoints
    i += 1
    return color_image, prev_descriptors, prev_keypoints, i, matches

