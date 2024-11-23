import cv2
import math
import mediapipe as mp
from pygame import mixer
import streamlit as st
import numpy as np

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


def find_angle(p1, p2, ref_pt = np.array([0,0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    degree = int(180 / np.pi) * theta

    if p1[0] < ref_pt[0]:
        degree = -degree
    
    return degree

