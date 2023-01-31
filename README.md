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
- **stop**, it stops the node from publishing any results. If "visualization" is set to true, it will close the visualization window;
- **start**, restarts publishing and relaunches the visualization window if the "visualization" is set to true;
- **reset**, resets the node using default options;

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
- **take_photo**, it takes a photo of the person or persons currently being detected;
- **enable_automatic**, it activates the automatic mode where for each new person detected, it automatically takes a photo and adds them to the detection record;
- **disable_automatic**, it deactivates the automatic mode;
- **stop**, it stops the node from publishing any results. If "visualization" is set to true, it will close the visualization window;
- **start**, restarts publishing and relaunches the visualization window if the "visualization" is set to true;
- **reset**, resets the node using default options;

The node also publishes other information regarding the person or persons detected and the detection record. The topics are the following:

```bash
/perception/reid/current_detection
/perception/reid/detection_record
```

Both topics are published using a custom message (StringArray.msg). If the node does not recognize a person, it will display "Unknown". The detection record only keeps track of people whose photo was taken and added to the encoder.


## Perception API (perception.py)
This node requires the mediapipe holistic node to be running and the Detectron or YOLO nodes. The first provides the measurements needed for the pointing direction as an example, and the second the information regarding object detection.

The API has the follwing actions:

- detectPointingObject(yolo, easyDetection, useFilteredObjects, classNameToBeDetected)

  It takes four parameters as inputs. The first three are boolean (True or False), and the last is a string. The yolo parameter tells the node to subscribe to the YOLO or to the Detectron results. 
  
  The easyDetection stands for two types of detection. The first, the simple approach, focuses on the arm direction to determine the pointing direction. Then it selects the object farthest left or farthest right accordingly. This approach works under the assumption that the useFilteredObjects is also set to true and that we are choosing between two objects. 
  The second approach finds which object gets intercepted by the pointing line segment and returns that object. If no object is detected, it returns the one closest to the line.
  
  When the useFilteredObjects input parameter is true, the node will look at objects whose class is given by the classNameToBeDetected input parameter. 

  ```bash
    yolo = True
    easyDetection = False
    useFilteredObjects = True
    classNameToBeDetected = "backpack"
    
    node_name = "perception_action"
    rospy.init_node(node_name, anonymous=True)
    rospy.loginfo("%s node created" % node_name)

    n_percep = Perception(yolo, easyDetection, useFilteredObjects, classNameToBeDetected)
    obj = n_percep.detectPointingObject()
  ```