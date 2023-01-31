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