import cv2
import os
import numpy as np
import streamlit as st
import csv
import time
from datetime import datetime

def take_attendance():
    """
    Performs live face recognition and records attendance.
    """
    st.header("Take Attendance")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = "trainer.yml" # Add variable to path

    # 1. Check if the model file exists
    if not os.path.exists(model_path):
        st.error(f"Error: The model file '{model_path}' does not exist.")
        st.error("Make sure you have trained the model first by running the 'Add Face' module.")
        return

    try:
        recognizer.read(model_path)  # Load the trained model
        st.success(f"Model '{model_path}' loaded successfully.") # Success statement

    except cv2.error as e:
        st.error(f"OpenCV Error loading '{model_path}': {e}")
        st.error("Make sure the file is a valid OpenCV model file and that the OpenCV version is compatible.")
        return
    except Exception as e:
        st.error(f"General Error loading '{model_path}': {e}")
        st.error("Check file permissions and ensure that the file is not corrupted.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load id_map from file
    id_map = {}
    id_map_path = "id_map.txt"

    if not os.path.exists(id_map_path):
        st.error(f"Error: The ID map file '{id_map_path}' does not exist.")
        st.error("Make sure you have trained the model first by running the 'Add Face' module.")
        return

    try:
        with open(id_map_path, "r") as f:
            for line in f:
                name, id = line.strip().split(":")
                id_map[int(id)] = name  # Reverse the mapping for prediction
        st.success(f"ID map '{id_map_path}' loaded successfully") #Sucess Statement
    except FileNotFoundError:
        st.error("id_map.txt not found. Make sure you have trained the model first by running the 'Add Face' module.")
        return
    except Exception as e:
        st.error(f"Error loading id_map.txt: {e}")
        return

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Error: Could not access the webcam.")
        return

    COL_NAMES = ['NAME', 'TIME']
    attendance = [] #To load each image to array

    frame_placeholder = st.empty()

    while True:
        ret, frame = video_capture.read()

        if not ret:
            st.error("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        names = []  # List to store detected names
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]

            try:
                id_, confidence = recognizer.predict(roi_gray)
                confidence = 100 - confidence  # Flip for easier interpretation
            except Exception as e:
                st.warning(f"Error during prediction: {e}")
                continue  # Skip to the next face

            if confidence > 50:  # Adjust confidence threshold as needed
                if id_ in id_map:  # Make sure ID exists in our id_map
                    name = id_map[id_]
                else:
                    name = "Unknown"  # ID not found

                cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                names.append(name)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                names.append("Unknown")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR", caption="Live Face Recognition") #Show images

        # Attendance Code
        k = cv2.waitKey(1)
        if k == ord('o'):  # Press 'o' to record the face
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            exist = os.path.isfile("attendance/Attendance_" + date + ".csv")
            if len(names) > 0:
                with open("attendance/Attendance_" + date + ".csv", "a", newline="") as csvfile:  # Use "a" for append mode
                    writer = csv.writer(csvfile)
                    if exist:
                        for name in names:
                            writer.writerow([name, timestamp])
                    else:
                        writer.writerow(COL_NAMES)
                        for name in names:
                            writer.writerow([name, timestamp])

                st.success("Attendance Taken..")
            else:
                st.warning("No Faces Detected in the Frame")

        if k == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()