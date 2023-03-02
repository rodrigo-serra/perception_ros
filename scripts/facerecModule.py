import numpy as np
import cv2
import face_recognition


# Inspired in https://github.com/ageitgey/face_recognition
def drawRectangleAroundFace(frame, current_detection, face_boundary, offset):
    # Display the results
    for d in current_detection:
        if face_boundary:
            top = d.top + offset
            bottom = d.bottom - offset
            left = d.left + offset
            right = d.right - offset
        else:
            top = d.top
            bottom = d.bottom
            left = d.left
            right = d.right
        
        name = d.id
        gender = d.gender
        age = d.ageRange

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # cv2.rectangle(frame, (left, bottom + 40), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, gender, (left + 6, bottom + 40), font, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, age, (left + 6, bottom + 60), font, 0.6, (0, 0, 255), 1)
    
    return frame


def faceRecognition(frame, known_face_encodings, known_face_names, use_distance = False):
    
    face_locations = []
    face_encodings = []
    face_names = []
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if ~use_distance:
            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
        else:
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances != []:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

        face_names.append(name)
        # print(name)

    return face_locations, face_names