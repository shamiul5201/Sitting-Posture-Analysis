import cv2
import math
import mediapipe as mp
from pygame import mixer
import streamlit as st
import numpy as np


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


camera_aligned = False

# =============================CONSTANTS and INITIALIZATIONS=====================================#

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors in RGB format.
blue = (0, 127, 255)
red = (255, 50, 50)
green = (0, 255, 127)
dark_blue = (0, 20, 127)
light_green = (100, 233, 127)
yellow = (255, 255, 0)
pink = (255, 0, 255)
magenta = (0, 255, 255)
white = (255,255,255)

# Initialize mediapipe pose class.
mp_pose, pose = get_mediapipe_pose()

# Initialize the sound alarm object.
alarm_file_path = r"beep.wav"
mixer.init()
play_sound = mixer.Sound(alarm_file_path)
# ===============================================================================================#

# To release camera resources when a state change (button click, value change, etc.) occurs.
# Streamlit re reuns the app from top to bottom and the variables are lost but the camera resource is not released
# This is indicated by the camera light still on (sometimes even if the tab is closed).

try:
    cap = st.session_state["camera_cap"]
    if cap is not None:
        cap.release()

    st.session_state["camera_cap"] = None
except KeyError:
    cap = None


st.markdown("<h1 style='text-align: center; color: #003380;'>Sitting Posture Analysis</h1>", unsafe_allow_html=True)


camera_dict = get_camera_dict()

col1, col2 = st.columns([3,1])

with col1:
    selected_option  = st.selectbox("Select Camera", options= sorted(camera_dict.keys()))


dict_features = {}
left_features  = {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                         'hip': mp_pose.PoseLandmark.LEFT_HIP,
                         'ear': mp_pose.PoseLandmark.LEFT_EAR,
                 }
right_features = {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                         'hip': mp_pose.PoseLandmark.RIGHT_HIP,
                         'ear': mp_pose.PoseLandmark.RIGHT_EAR,
                 }

dict_features['left'] = left_features
dict_features['right'] = right_features
dict_features['nose'] = mp_pose.PoseLandmark.NOSE


if selected_option:

    # Collect frames to display
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(camera_dict[selected_option])
    st.session_state["camera_cap"] = cap

    with col2:
        m = st.empty()
        m.markdown(
        """
        <style>
        div.stButton > button:first-child {
                background: #76b900 ;
                color: #fff !important;
                border-radius: 5px ;
                font-size: 21px ;
                padding: 5px 25px ;
                margin-top: 12px ;
                border:1px solid #76b900 ;
                box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19) ;
        }
        div.stButton > button:hover {
                    background-color:#3e8e41;         

        }
        </style>
        """,
        unsafe_allow_html=True,
        )

        # Click if you want to run the application.
        run_btn_container = st.empty()
        run = run_btn_container.button("START")

    while cap.isOpened():
            # Capture frames.
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.putText(frame, selected_option, (10, 30), font, 0.9, magenta, 2)

            FRAME_WINDOW.image(frame)

            if run or not success:
                break

    FRAME_WINDOW.empty()                
    cap.release()
    cv2.destroyAllWindows()


    OFFSET_THRESH = st.sidebar.slider("Offset Threshold:", 0.0, 40.0, 30.0, 10.0)

    NECK_THRESH_POS  = st.sidebar.slider("Neck Angle Thresh (+):", 0.0, 40.0, 35.0, 5.0)
    NECK_THRESH_NEG  = st.sidebar.slider("Neck Angle Thresh (-):", -20.0, 0.0, -5.0, 5.0)

    TORSO_THRESH_POS = st.sidebar.slider("Torso Angle Thresh (+):", 0.0, 20.0, 10.0, 5.0)
    TORSO_THRESH_NEG = st.sidebar.slider("Torso Angle Thresh (-):", -20.0, 0.0, -5.0, 5.0)

    BAD_TIME = st.sidebar.slider("Seconds to wait before sounding alarm:", 0.0, 20.0, 5.0, 0.5)


    if run:

        cap = cv2.VideoCapture(camera_dict[selected_option])

        # Get the FPS.
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        st.session_state["camera_cap"] = cap
        
        m.empty()
        m.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background: #dd0000 ;
            color: #fff !important ;
            border-radius: 5px ;
            font-size: 21px ;
            padding: 5px 25px ;
            margin-top: 12px ;
            border:1px solid #dd0000 ;
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
            
        }
        div.stButton > button:hover {
                    background-color:#800000;         

        }
        </style>
        """,
        unsafe_allow_html=True,
       )
        # Change button text.
        run_btn_container.empty()
        run_btn_container.button("STOP")

        bad_time = 0
        good_time = 0
        t1 = cv2.getTickCount()
        while cap.isOpened():
            # Capture frames and start timer

            success, image = cap.read()

            if not success:
                print("Null Frames")
                break

            # Get fps.
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Get height and width.
            h, w = image.shape[:2]

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            # Process the image.
            keypoints = pose.process(image)

            # Use lm and lmPose as representative of the following methods.
            if keypoints.pose_landmarks:
            
                ps_lm = keypoints.pose_landmarks

                nose_coord = get_landmark_features(ps_lm.landmark, 'nose', w, h)
                left_shldr_coord, left_ear_coord, left_hip_coord = get_landmark_features(ps_lm.landmark, 'left', w, h)
                right_shldr_coord, right_ear_coord, right_hip_coord = get_landmark_features(ps_lm.landmark, 'right', w, h)

                offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

                if abs(offset_angle) > OFFSET_THRESH:

                    cv2.circle(image, nose_coord, 7, white, -1)
                    cv2.circle(image, left_shldr_coord, 7, yellow, -1)
                    cv2.circle(image, right_shldr_coord, 7, pink, -1)
                    cv2.putText(image, 'CAMERA NOT ALIGNED PROPERLY!!!', (10, h-40), font, 0.9, red, 2)
                    cv2.putText(image, 'OFFSET ANGLE: '+str(int(abs(offset_angle))), (10, h-5), font, 0.9, red, 2)

                # Camera is aligned properly.
                else:

                    cv2.putText(image, str(int(abs(offset_angle))) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)

                    dist_l_sh_hip = np.linalg.norm(left_hip_coord - left_shldr_coord)
                    dist_r_sh_hip = np.linalg.norm(right_hip_coord - right_shldr_coord)

                    if dist_l_sh_hip > dist_r_sh_hip:

                        neck_inclination = -find_angle(left_ear_coord, np.array([left_shldr_coord[0], 0]), left_shldr_coord)
                        torso_inclination = -find_angle(left_shldr_coord, np.array([left_hip_coord[0], 0]), left_hip_coord)

                        shldr_coord = left_shldr_coord
                        ear_coord = left_ear_coord
                        hip_coord = left_hip_coord

                        cv2.putText(image, 'Using Left Side', (10, h - 50), font, 0.9, green, 2)

                    
                    else:
                        
                        neck_inclination = find_angle(right_ear_coord, np.array([right_shldr_coord[0], 0]), right_shldr_coord)
                        torso_inclination = find_angle(right_shldr_coord, np.array([right_hip_coord[0], 0]), right_hip_coord)

                        shldr_coord = right_shldr_coord
                        ear_coord = right_ear_coord
                        hip_coord = right_hip_coord

                        cv2.putText(image, 'Using Right Side', (10, h - 50), font, 0.9, green, 2)

                    
                    # Draw landmarks.
                    cv2.circle(image, shldr_coord, 7, yellow, -1)
                    cv2.circle(image, ear_coord, 7, yellow, -1)
                    cv2.circle(image, hip_coord, 7, yellow, -1)

                    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
                    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
                    cv2.circle(image, (shldr_coord[0], shldr_coord[1] - 100), 7, yellow, -1)

                    # Similarly, here we are taking y - coordinate 100px above x1. Note that
                    # you can take any value for y, not necessarily 100 or 200 pixels.
                    cv2.circle(image, (hip_coord[0], hip_coord[1] - 100), 7, yellow, -1)
                    

                    # Put text, Posture and angle inclination.
                    # Text string for display.
                    neck_text_string  = 'Neck : ' + str(int(neck_inclination))
                    torso_text_string = 'Torso : ' + str(int(torso_inclination))

                    # Determine whether good posture or bad posture.
                    # The threshold angles have been set based on intuition.
                    if (neck_inclination <= NECK_THRESH_POS and neck_inclination >= NECK_THRESH_NEG)  and (torso_inclination <= TORSO_THRESH_POS and torso_inclination >= TORSO_THRESH_NEG):
                        bad_time = 0
                        t2 = cv2.getTickCount()
                        good_time += (t2 - t1) / cv2.getTickFrequency()
                        
                        cv2.putText(image, neck_text_string, (10, 30), font, 0.9, light_green, 2)
                        cv2.putText(image, torso_text_string, (200, 30), font, 0.9, light_green, 2)
                        cv2.putText(image, str(int(neck_inclination)), (shldr_coord[0] + 10, shldr_coord[1]), font, 0.9, light_green, 2)
                        cv2.putText(image, str(int(torso_inclination)), (hip_coord[0] + 10, hip_coord[1]), font, 0.9, light_green, 2)

                        # Join landmarks.
                        cv2.line(image, shldr_coord, ear_coord, green, 4)
                        cv2.line(image, shldr_coord, (shldr_coord[0], shldr_coord[1] - 100), green, 4)
                        cv2.line(image, hip_coord, shldr_coord, green, 4)
                        cv2.line(image, hip_coord, (hip_coord[0], hip_coord[1] - 100), green, 4)

                    else:
                        good_time = 0
                        t2 = cv2.getTickCount()
                        bad_time += (t2 - t1) / cv2.getTickFrequency()

                        neck_color = red
                        torso_color = red

                        if neck_inclination <= NECK_THRESH_POS and neck_inclination >= NECK_THRESH_NEG:
                            neck_color = light_green
                        
                        if torso_inclination <= TORSO_THRESH_POS and torso_inclination >= TORSO_THRESH_NEG:
                            torso_color = light_green

                        cv2.putText(image, neck_text_string, (10, 30), font, 0.9, neck_color, 2)
                        cv2.putText(image, torso_text_string, (200, 30), font, 0.9, torso_color, 2)
                        cv2.putText(image, str(int(neck_inclination)), (shldr_coord[0] + 10, shldr_coord[1]), font, 0.9, red, 2)
                        cv2.putText(image, str(int(torso_inclination)), (hip_coord[0] + 10, hip_coord[1]), font, 0.9, red, 2)

                        # Join landmarks.
                        cv2.line(image, shldr_coord, ear_coord, red, 4)
                        cv2.line(image, shldr_coord, (shldr_coord[0], shldr_coord[1] - 100), red, 4)
                        cv2.line(image, hip_coord, shldr_coord, red, 4)
                        cv2.line(image, hip_coord, (hip_coord[0], hip_coord[1] - 100), red, 4)
                        

                    t1 = t2

                    if good_time > 0:
                        time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
                        
                    elif bad_time > 0:
                        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

                    if bad_time>=BAD_TIME:
                            play_sound.play()
                    
            
            FRAME_WINDOW.image(image)
    
            
    FRAME_WINDOW.empty()

    cap.release()
    cv2.destroyAllWindows()

