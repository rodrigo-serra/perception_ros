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
It's launched by the mediapipe_holistic.launch where all the variables are set. This node also depends on the holisticDetectorModule.py where all the operations regarding mediapipe take place. In this module, there are some threshold parameters, such as the landmark visibility threshold and the hand distance to body threshold. These variables should be of no concern but can be tuned if needed.

Regarding topics, the node subscribes to the topic **event_in** which can take a string msg. Depending on the msg, the node will behave differently. The options are the following:
