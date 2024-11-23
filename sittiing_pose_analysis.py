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

    