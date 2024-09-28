import cv2
import numpy as np
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | 
                                         PyKinectV2.FrameSourceTypes_Depth)

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Main loop
frame_count = 0
faces = []
while True:
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        frame = np.reshape(color_frame, (kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))
        frame = frame[:, :, :3]  # Drop alpha channel
        frame = cv2.convertScaleAbs(frame)  # Convert to 8-bit unsigned int

        frame_count += 1
        if frame_count % 5 == 0:  # Detect faces every 5 frames
            faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Kinect Face Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()