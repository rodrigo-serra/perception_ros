# Perception Reid

## Dependecies
```bash
pip3 install -r requirements.txt --upgrade
```
&nbsp;

## Mediapipe Holistic
### **mediapipe_holistic.launch**
It launches the mediapipe holistic node (mediapipeHolisticnode.py) and sets all the variables required for the proper functioning of the node. The variables are the following:

- "camera_topic"
```bash
<arg name="camera_topic" default="/object_detector/detection_image/compressed" />
```

- "img_compressed": It will depend on the "camera_topic". Set it to true or false accordingly.
```bash
<arg name="img_compressed" default="true" />
```

- "visualization": If set to true, the node will launch a visualization window.
```bash
<arg name="visualization" default="true" />
```

- "draw...": If "visualization" is set to true, the node will illustrate the landmarks depending on their boolean value. If "drawFaceBoundary" is set to true, the visualization will only display the person's face."
```bash
  <arg name="drawPoseLandmarks" default="true" />
  <arg name="drawFaceLandmarks" default="false" />
  <arg name="drawRightHandLandmarks" default="true" />
  <arg name="drawLeftHandLandmarks" default="true" />
  <arg name="drawFaceBoundary" default="false" />
```

- "pointing_hands": If set to true, the node will use the hand landmarks to determine the pointing direction. Otherwise it will use the body landmarks (elbow and wrist)"
```bash
 <arg name="pointing_hands" default="false" />
```

- "pointing_right_hand_msg": Message definition for pointing direction.
```bash
 <arg name="pointing_right_hand_msg" default="right" />
 <arg name="pointing_left_hand_msg" default="left" />
```

### **mediapipeHolisticnode.py**
It's launched by the mediapipe_holistic.launch where all the variables are set. This node also depends on the holisticDetectorModule.py where all the operations regarding mediapipe take place. In this module, there are some threshold parameters, such as the landmark visibility threshold and the hand distance to body threshold. 

The first determines if a landmark is of any use, i.e., if the landmark visibility score is below the threshold, then we should not trust the measurement. The hand distance to the body threshold determines if we compute or not the pointing direction. 
These variables should be of no concern but can be tuned if needed.

Regarding topics, the node subscribes to the topic **event_in** which can take a string msg. 
```bash
/perception/mediapipe_holistic/event_in
```

Depending on the msg, the node will behave differently. The options are the following:
- **e_stop**, it stops the node from publishing any results. If "visualization" is set to true, it will close the visualization window;
- **e_start**, restarts publishing and relaunches the visualization window if the "visualization" is set to true;
- **e_reset**, resets the node using default options;

The node also publishes the mediapipe holistic results and some extra information one can extract from the latter. The topics are the following:

```bash
/perception/mediapipe_holistic/face_landmarks
/perception/mediapipe_holistic/pose_world_landmarks
/perception/mediapipe_holistic/img_pose_landmarks
/perception/mediapipe_holistic/right_hand_landmarks
/perception/mediapipe_holistic/left_hand_landmarks

/perception/mediapipe_holistic/right_arm_length
/perception/mediapipe_holistic/left_arm_length
/perception/mediapipe_holistic/shoulder_length
/perception/mediapipe_holistic/hip_length
/perception/mediapipe_holistic/torso_length

/perception/mediapipe_holistic/hand_pointing_direction
/perception/mediapipe_holistic/hand_pointing_intercept
/perception/mediapipe_holistic/hand_pointing_slope

/perception/mediapipe_holistic/sweater_color
```

The topics with "landmarks" on its name return the 2D or 3D coordinates for each landmark given by the mediapipe holistic library. The data is published using a custom msg type called MediapipePointInfoArray. We recommend going through the mediapipe documentation <a href="https://google.github.io/mediapipe/solutions/holistic.html" target="_blank">mediapipe documentation</a> to understand the whole landmark structure.

With the measurements we get from mediapipe, we can get information such as right and left arm length, shoulder length, and so on. Topics with length on their name carry this extra data in a Float32 message format. These measurements can be added as required. For instance, the right and left leg lengths aren't published yet.

This node also determines which hand someone is pointing with and computes the slope and intercept of the line segment associated with the pointing direction. The ".../hand_pointing_slope" and the ".../hand_pointing_intercept" topics are published in a Float32 message format whereas the ".../hand_pointing_direction" topic publishes a string.

&nbsp;

## Reid
### **reid.launch**
It launches the reid node (reidnode.py) and sets all the variables required for the proper functioning of the node. The variables are the following:

- "camera_topic"
```bash
<arg name="camera_topic" default="/camera/color/image_raw" />
```

- "img_compressed": It will depend on the "camera_topic". Set it to true or false accordingly.
```bash
<arg name="img_compressed" default="false" />
```

- "visualization": If set to true, the node will launch a visualization window.
```bash
<arg name="visualization" default="true" />
```

- "extract_face_boundary_only": If set to true, the node will use the mediapipe holistic module to extract the face boundary. Otherwise, it will take a rectangular photo of the person's face.
```bash
<arg name="extract_face_boundary_only" default="true" />
```

### **reidnode.py**
It's launched by the reid.launch where all the variables are set. This node also depends on the holisticDetectorModule.py where all the operations regarding mediapipe take place, and on the facerecModule.py where the reid is performed. 

Regarding topics, the node subscribes to the topic **event_in** which can take a string msg. 
```bash
/perception/reid/event_in
```

Depending on the msg, the node will behave differently. The options are the following:
- **e_take_photo**, it takes a photo of the person or persons currently being detected;
- **e_enable_automatic**, it activates the automatic mode where for each new person detected, it automatically takes a photo and adds them to the detection record;
- **e_disable_automatic**, it deactivates the automatic mode;
- **e_stop**, it stops the node from publishing any results. If "visualization" is set to true, it will close the visualization window;
- **e_start**, restarts publishing and relaunches the visualization window if the "visualization" is set to true;
- **e_reset**, resets the node using default options;

The node also publishes other information regarding the person or persons detected and the detection record. The topics are the following:

```bash
/perception/reid/current_detection
/perception/reid/detection_record
```

Both topics are published using a custom message (ReidInfoArray.msg). If the node does not recognize a person, it will display "Unknown". It will also estimate the gender and the age range of the person being detected. The detection record only keeps track of people whose photo was taken and added to the encoder.


## Perception API (perception.py)


The API has the follwing actions:

- detectPointingObject(useYolo = False, easyDetection = False, useFilteredObjects = True, classNameToBeDetected = 'backpack', score = 0.5)
  
  This action returns the object someone is pointing at. It requires the mediapipe holistic node to be running and the Detectron or YOLO nodes. The first provides the measurements needed for the pointing direction, and the second the information regarding object detection. The msg type is RecognizedObject.

  It takes five parameters as inputs. The yolo parameter tells the node to subscribe to the YOLO or to the Detectron results. 
  
  The easyDetection stands for two types of detection. The first, the simple approach, focuses on the arm direction to determine the pointing direction. Then it selects the object farthest left or farthest right accordingly. This approach works under the assumption that the useFilteredObjects is also set to true and that we are choosing between two objects. 
  The second approach finds which object gets intercepted by the pointing line segment and returns that object. If no object is detected, it returns the one closest to the line.
  
  When the useFilteredObjects input parameter is true, the node will look at objects whose class is given by the classNameToBeDetected input parameter and whose score (confidence) is above the threshold. 


- detectPointingObjectWithCustomMsg(easyDetection = False, useFilteredObjects = True, classNameToBeDetected = 'backpack', score = 0.5)
  
  This action returns the object someone is pointing at + the corresponding depth img. It requires the mediapipe holistic node to be running and the Detectron node. The msg type is SingleRecognizedObjectWithMask.

  It takes four parameters as inputs.
  
  The easyDetection stands for two types of detection. The first, the simple approach, focuses on the arm direction to determine the pointing direction. Then it selects the object farthest left or farthest right accordingly. This approach works under the assumption that the useFilteredObjects is also set to true and that we are choosing between two objects. 
  The second approach finds which object gets intercepted by the pointing line segment and returns that object. If no object is detected, it returns the one closest to the line.
  
  When the useFilteredObjects input parameter is true, the node will look at objects whose class is given by the classNameToBeDetected input parameter and whose score (confidence) is above the threshold. 

  Note that this action takes a bit longer to return an output since it waits for the synchronization of the depth and detectron topics.


- returnDetectedObjects(useYolo = False, useFilteredObjects = True, classNameToBeDetected = 'backpack', score = 0.5)
  
  It returns all the objects detected by the YOLO or the detectron. Hence, it requires one of the nodes to be running. The msg type is RecognizedObjectArrayStamped.

  It takes four parameters as inputs. The yolo parameter tells the node to subscribe to the YOLO or to the Detectron results. 

  When the useFilteredObjects input parameter is true, the node will look at objects whose class is given by the classNameToBeDetected input parameter and whose score (confidence) is above the threshold. 


- getObjectNames()

  It returns the detected objects class names. 


- getPointingDirection()

  It returns the pointing direction, i.e., left or right. The msg type is Float32. It requires the mediapipe holistic node to be running.

- getPointingSlope()

  It returns the slope of the pointing line segment. The msg type is Float32. It requires the mediapipe holistic node to be running.

- getPointingIntercept()

  It returns the intercept of the pointing line segment. The msg type is Float32. It requires the mediapipe holistic node to be running.

- getPoseWorldLandmarks()
  
  It returns the 3D coordinates (x, y, z) and score (visibility) of each body landmark. The msg type is MediapipePointInfoArray. It requires the mediapipe holistic node to be running.

- getFaceLandmarks()

  It returns 2D coordinates (x, y) and the score (visibility) for each face landmark. Note that it will also return a z coordinate which is -1, thus ought to be ignored. The msg type is MediapipePointInfoArray. It requires the mediapipe holistic node to be running.

- getImgPoseLandmarks()

  It returns 2D coordinates (x, y) and the score (visibility) for each img pose landmark. Note that it will also return a z coordinate which is -1, thus ought to be ignored. The msg type is MediapipePointInfoArray. It requires the mediapipe holistic node to be running.


- getRightHandLandmarks()

  It returns 2D coordinates (x, y) for each img right hand landmark. Note that it will also return a z coordinate which is -1 and score (visibility) also -1, thus ought to be ignored. The msg type is MediapipePointInfoArray. It requires the mediapipe holistic node to be running.

- getLeftHandLandmarks()

  It returns 2D coordinates (x, y) for each img left hand landmark. Note that it will also return a z coordinate which is -1 and score (visibility) also -1, thus ought to be ignored. The msg type is MediapipePointInfoArray. It requires the mediapipe holistic node to be running.

<!--
- getHipLength()

  It returns the hip length. The msg type is Float32. It requires the mediapipe holistic node to be running.

- getTorsoLength()

  It returns the torso length. The msg type is Float32. It requires the mediapipe holistic node to be running.

- getShoulderLength()

  It returns the shouler length. The msg type is Float32. It requires the mediapipe holistic node to be running.

- getRightArmLength()

  It returns the right arm length. The msg type is Float32. It requires the mediapipe holistic node to be running.

- getLeftArmLength()

  It returns the left arm length. The msg type is Float32. It requires the mediapipe holistic node to be running.

- readSweaterColor()

  It returns the estimated color of the person's sweater/t-shirt.
-->

- getPeopleDetection()

  It returns an array with all the persons detected in the current frame. The msg type is ReidInfoArray. It requires the reid node to be running.

- getPeopleDetectionRecord()

  It returns an array with all the persons detected in the past. The msg type is ReidInfoArray. It requires the reid node to be running.

- startReid()

  It starts the Reid node. It requires the reid node to be running.

- stopReid()

  It stops the Reid node. It requires the reid node to be running.

- resetReid()

  It resets the Reid node. It requires the reid node to be running.

- takePhotoReid()

  Meant to take a photo of a person and added to the detection record. It requires the reid node to be running.


- enableAutomaticReid()

  Activates automatic reid, i.e., every time a new person is detected, the node will capture a photo and added it to the detection record. It requires the reid node to be running.


- disableAutomaticReid()

  Disables automatic reid. It requires the reid node to be running.


 - startMediapipeHolistic()

    It starts the Mediapipe holistic node. It requires the mediapipe holistic node to be running.


 - stopMediapipeHolistic()

    It stops the Mediapipe holistic node. It requires the mediapipe holistic node to be running.


 - resetMediapipeHolistic()


    It resets the Mediapipe holistic node. It requires the mediapipe holistic node to be running.


 - startDetectron()


    It starts the detectron node and it loads the model. It requires the detectron node to be running.


 - startDetectronTopics()


    The node starts subscribing and publishing. It requires the detectron node to be running.


 - stopDetectron()


    It stops the detectron node and it deletes the model. It requires the detectron node to be running.


 - stopDetectronTopics()


    The node stops subscribing and publishing. It requires the detectron node to be running.




