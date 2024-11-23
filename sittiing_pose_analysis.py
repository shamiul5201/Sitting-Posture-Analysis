import cv2
import math
import mediapipe as mp
from pygame import mixer
import streamlit as st
import numpy as np


dict_features = {
    'nose': 0,  # Replace with the index for the nose landmark
    'left': {
        'shoulder': 12,  
        'ear': 7,      
        'hip': 24       
    },
    'right': {
        'shoulder': 11,  
        'ear': 8,       
        'hip': 23       
    }
}


def get_camera_dict():
    index = 0
    cam_dict = {}
    i = 5
    count = 0

    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cam_dict['camera'+str(count + 1)] = index
            cap.release()
            count+=1
        index += 1
        i -= 1

    cam_dict['Camera 1 (default)'] = cam_dict.pop('Camera 1')
    return cam_dict



def find_angle(p1, p2, ref_pt = np.array([0,0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref,p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
    degree = int(180 / np.pi) * theta

    if p1[0] < ref_pt[0]:
        degree = -degree

    return degree

def get_landmark_array(pose_landmark, key, frame_width, frame_height):

    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)

    return np.array([denorm_x, denorm_y])


def get_landmark_features(kp_results, feature, frame_width, frame_height):

    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        ear_coord   = get_landmark_array(kp_results, dict_features[feature]['ear'], frame_width, frame_height)
        hip_coord   = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)

        return shldr_coord, ear_coord, hip_coord
    
    else:
       raise ValueError("feature needs to be either 'nose', 'left' or 'right")


@st.cache(allow_output_mutation=True)
def get_mediapipe_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    return mp_pose, pose
camera_aligned = True

