import cv2
import numpy as np
import os
from time import sleep
import time
from datetime import datetime


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../trainer/trainer.yml')
cascadePath = "../cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
# initiate id counter
id = 1
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['none', 'austin', 'anne', 'friend1', 'friend2', 'jane', 'Brian', 'George', 'Nathan', 'Ken']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=1,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (110, 37, 47), 2)
        cv2.rectangle(img, (x, y - 45), (x + w, y), (110, 37, 47), cv2.FILLED)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print(x, y, w, h)
        coords = (x, y, w, h)
        # If confidence is less them 100 ==> "0" : perfect match 
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(id).upper(),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )
        print(id)

        nowtime = time.strftime("%H:%M:%S:%MS", time.localtime())
        def savefacedata(id):
            passpath = 'facedetectiondata.csv'
            try:
                open(passpath, 'x+')
                print('csv file', passpath, 'created')
            except FileExistsError:
                print('csv file', passpath, 'already exists')
            with open('facedetectiondata.csv', 'a+') as f:
                f.writelines(f'\n{nowtime},{id},{coords}')


        savefacedata(id)
    cv2.imshow('face recognizer', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27 or k == 113:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
print('code completed')
