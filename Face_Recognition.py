import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

KNOWN_FACES_DIR = "known_faces"
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    img = cv2.imread(f"{KNOWN_FACES_DIR}/{name}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        known_faces.append(img_gray[y:y+h, x:x+w])
        known_names.append(name.split(".")[0])

attendance = pd.DataFrame(columns=["Name", "Time"])

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        for i, known_face in enumerate(known_faces):
            if np.array_equal(face, known_face):
                name = known_names[i]
                time = datetime.now().strftime('%H:%M:%S')
                if name not in attendance['Name'].values:
                    attendance = attendance.append({"Name": name, "Time": time}, ignore_index=True)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
attendance.to_excel("attendance.xlsx", index=False)
