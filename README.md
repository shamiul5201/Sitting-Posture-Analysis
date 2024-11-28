## Sitting-Posture-Analysis
This documentation provides an in-depth overview of a Python-based application for analyzing sitting posture using MediaPipe, OpenCV, Streamlit, and Pygame. It leverages computer vision and pose estimation to determine whether a user's posture is aligned correctly and provides feedback accordingly. Below is a breakdown of the code and its components.

### Dependencies
The application requires the following libraries:

* **OpenCV**: For video capture and image processing.
* **Math**: For angle calculations.
* **MediaPipe**: For pose detection.
* **Pygame**: To play audio alerts.
* **Streamlit**: For the user interface.
* **NumPy**: For numerical operations.

## Mediapipe Pose
MediaPipe Pose is a machine learning solution for high-fidelity body pose tracking that infers 33 3D landmarks and a background segmentation mask on the entire body from RGB video frames using BlazePose research, which also drives the ML Kit Pose Detection API.

**Scenary of The Points from Mediapipe Pose:**
![77738pose_tracking_full_body_landmarks](https://github.com/user-attachments/assets/26d2cb7b-eecb-4cd2-ab91-fe1da617a692)


Below is the workflow of our Body Posture Detection System


![workflow](https://github.com/user-attachments/assets/9692d790-31e1-471d-b5ad-5212e3fbf03a)


## Core Functions
### Camera Selection
at first, imported the required libraries which are:
```python
def get_camera_dict():
    ...

```
Detects and lists available cameras (up to 5) on the system, assigning them unique labels (e.g., "Camera 1").

Input: None.
Output: A dictionary mapping camera labels to their respective indices.
Key Feature: Defaults to the first camera in the list.

### Angle Calculation

The setup requires the individual to be positioned in the correct side view. The function find_angle determines the offset angle between three points, such as the hips, eyes, or shoulders, which are generally symmetric about the central axis of the human body.
![angle](https://github.com/user-attachments/assets/6925702c-b4b5-4e94-8a3e-e7f2700b7e81)

The angle between three points, with one serving as the reference point, is calculated using the following equation:

$$
\theta = \arccos \left( \frac{\overrightarrow{P_1 ref} \cdot \overrightarrow{P_2 ref}}{|\overrightarrow{P_1 ref}| \cdot |\overrightarrow{P_2 ref}|} \right)
$$

```python
def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    ...

```
Calculates the angle between two points relative to a reference point.

Input: Two points `(p1, p2)` and a reference point `(ref_pt)`.
Output: The angle in degrees.

### Landmark Handling
* **`get_landmark_array`**: Converts normalized MediaPipe landmarks to pixel coordinates.
* **`get_landmark_features`**: Extracts specific landmarks (nose, shoulder, ear, hip) based on user-specified features.

### MediaPipe Pose Initialization
```python
@st.cache(allow_output_mutation=True)
def get_mediapipe_pose():
    ...
```
Initializes the MediaPipe Pose estimator to detect body landmarks.

## Streamlit Interface
The application uses **Streamlit** to create an interactive interface.

### Camera Selection
```python
camera_dict = get_camera_dict()
selected_option = st.selectbox("Select Camera", options=sorted(camera_dict.keys()))
```
Displays available cameras for the user to choose from.

### Threshold Controls
```python
OFFSET_THRESH = st.sidebar.slider("Offset Threshold:", 0.0, 40.0, 30.0, 10.0)
...

```
Allows users to adjust thresholds for:
* Offset angle.
* Neck and torso inclination angles.
* Time delay before triggering an alarm.
  
### Posture Monitoring
The live camera feed is displayed using:
```python
FRAME_WINDOW = st.image([])
```
A "START" button toggles posture analysis, while a "STOP" button halts it.


## Posture Analysis
### Landmark Detection

Landmarks for the nose, shoulders, ears, and hips are identified using MediaPipe.

### Camera Alignment
Checks the alignment of the camera based on the offset angle:
```python
if abs(offset_angle) > OFFSET_THRESH:
    ...
else:
    ...
```

### Posture Evaluation
* **Good Posture**: When neck and torso inclinations fall within the specified thresholds.
* **Bad Posture**: When any threshold is exceeded.
  
Visualization:
* Displays angle values and joins key landmarks with colored lines.
* Colors indicate posture correctness (e.g., green for good, red for bad).

## Alarms and Feedback
If a bad posture persists for a specified duration, an alarm sound is played:
```python
play_sound.play()
```

## Key Features
* **Real-time Posture Monitoring**: Detects and visualizes posture in live video.
* **Customizable Thresholds**: Users can tweak posture thresholds and response times.
* **Audio Feedback**: Alerts users when poor posture is detected.
* **Streamlit Integration**: Provides an intuitive web-based interface.

## Running the Application
1. Save the code in a file, e.g., posture_analysis.py.
2. Run the application with:
```python
streamlit run posture_analysis.py
```
3. Adjust camera settings and thresholds in the interface.
4. Start the posture analysis and observe feedback in real-time.
