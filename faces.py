import numpy as np
import cv2
import pickle

face = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
eye = cv2.CascadeClassifier("cascades/data/haarcascade_eye.xml")
smile = cv2.CascadeClassifier("cascades/data/haarcascade_smile.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizors/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    """ Capture frame by frame """
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, scaleFactor=1.5, minNeigbors=5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)
        stroke = 2 
        endCordX = x + w
        endCordy = y + h
        cv2.rectangle(frame, (x, y), (endCordX, endCordY), color, stroke)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()   
cv2.destroyAllWindows()



