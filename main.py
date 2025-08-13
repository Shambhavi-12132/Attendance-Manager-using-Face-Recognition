import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Debug: print current working directory
print("Current Working Directory:", os.getcwd())

# Load known faces (make sure these paths are correct!)
sham_image = face_recognition.load_image_file("C:\\Users\\shamb\\OneDrive\\Desktop\\python\\Projects.py\\Attendance\\faces\\sham.jpg")
sham_encoding = face_recognition.face_encodings(sham_image)[0]

ragini_image = face_recognition.load_image_file("C:\\Users\\shamb\\OneDrive\\Desktop\\python\\Projects.py\\Attendance\\faces\\ragini.jpg")
ragini_encoding = face_recognition.face_encodings(ragini_image)[0]

known_face_encodings = [sham_encoding, ragini_encoding]
known_face_names = ["sham", "ragini"]
students = known_face_names.copy()

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create/open CSV file for attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Resize frame and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = "Unknown"
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

        # Display name on the screen
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name + " Present", (10, 100), font, 1.5, (255, 0, 0), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Attendance", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
f.close()
