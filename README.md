## Sitting-Posture-Analysis
The secret to a person's general well-being is proper posture. However, frequently maintain proper body posture might be challenging. The actions necessary to create a solution for that is outlined in this repository. We will discover how to build a Sitting body position detection system using MediaPipe. Samples taken from the side will be used for analysis and conclusion-making. A webcam aimed at the person's side view would be necessary for a useful use. 

## Mediapipe Pose
MediaPipe Pose is a machine learning solution for high-fidelity body pose tracking that infers 33 3D landmarks and a background segmentation mask on the entire body from RGB video frames using BlazePose research, which also drives the ML Kit Pose Detection API.

**Scenary of The Points from Mediapipe Pose:**
![77738pose_tracking_full_body_landmarks](https://github.com/user-attachments/assets/26d2cb7b-eecb-4cd2-ab91-fe1da617a692)


Below is the workflow of our Body Posture Detection System


![workflow](https://github.com/user-attachments/assets/9692d790-31e1-471d-b5ad-5212e3fbf03a)


## Project Demonstration
at first, imported the required libraries which are:
```python
import cv2
import math
import mediapipe as mp
from pygame import mixer
import streamlit as st
import numpy as np
```
Before launching the app, obtaining a list of all available cameras is essential. From this list, the most suitable camera for capturing a good side view can be selected.
```python
def get_camera_dict():
    # checks the first 5 indexes.
    index = 0
    cam_dict = {}
    i = 5
    count = 0
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cam_dict['Camera '+str(count + 1)] =  index
            cap.release()
            count+=1
        index += 1
        i -= 1
    
    cam_dict['Camera 1 (default)'] = cam_dict.pop('Camera 1')
    return cam_dict
```
The setup requires the individual to be positioned in the correct side view. The function find_angle determines the offset angle between three points, such as the hips, eyes, or shoulders, which are generally symmetric about the central axis of the human body.
![angle](https://github.com/user-attachments/assets/6925702c-b4b5-4e94-8a3e-e7f2700b7e81)

The angle between three points, with one serving as the reference point, is calculated using the following equation:

$$
\theta = \arccos \left( \frac{\overrightarrow{P_1 ref} \cdot \overrightarrow{P_2 ref}}{|\overrightarrow{P_1 ref}| \cdot |\overrightarrow{P_2 ref}|} \right)
$$


**Coming soon**
