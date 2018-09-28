import cv2
import numpy as np
from PIL import Image
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
yLabels = []
xTrain = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id = label_ids[label]

            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")

            faces = face.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id)

with open("pickles/face-labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(xTrain, np.array(yLabels))
recognizer.save("recognizors/face-trainner.yml")



