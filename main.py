import cv2
import numpy as np
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | 
                                         PyKinectV2.FrameSourceTypes_Audio | 
                                         PyKinectV2.FrameSourceTypes_Depth)

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def get_audio_direction():
    audio_beam_angle = kinect._PyKinectRuntime__kinect.get_audio_beam_angle()
    return audio_beam_angle

def find_closest_face_to_audio(faces, beam_angle, frame_width):
    if beam_angle is None:
        return None
    
    audio_x = int((beam_angle + 1) / 2 * frame_width)  # Convert [-1, 1] to [0, frame_width]
    
    closest_face = None
    min_distance = float('inf')

    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        distance = abs(face_center_x - audio_x)
        if distance < min_distance:
            closest_face = (x, y, w, h)
            min_distance = distance
    
    return closest_face

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

        beam_angle = get_audio_direction()
        closest_face = find_closest_face_to_audio(faces, beam_angle, frame.shape[1])

        for (x, y, w, h) in faces:
            color = (0, 255, 0)  # Green for all faces
            if (x, y, w, h) == closest_face:
                color = (0, 0, 255)  # Red for the face closest to the audio source
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Visualize audio beam direction
        if beam_angle is not None:
            audio_x = int((beam_angle + 1) / 2 * frame.shape[1])  # Convert [-1, 1] to [0, frame_width]
            cv2.line(frame, (audio_x, 0), (audio_x, frame.shape[0]), (255, 0, 0), 2)

        cv2.imshow('Kinect Face Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()