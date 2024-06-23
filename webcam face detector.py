'''import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
#img=cv2.imread('image.jpg')
webcam = cv2.VideoCapture(0)
while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(randrange(256),randrange(256),randrange(256)),10)
    cv2.imshow('Austin face detector',frame)
    #quit when Q or q is pressed
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 81:
        break
webcam.release()
cv2.destroyAllWindows()
print('code completed')
import cv2
from random import randrange
def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )

# Capturing the Video Stream
video_capture = cv2.VideoCapture(0)

# Creating the cascade objects
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
#smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

while True:
    # Get individual frame
    _, frame = video_capture.read()
    # Covert the frame to grayscale
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
	# Detect all the faces in that frame
    detected_faces = face_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
    detected_eyes = eye_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
    #detected_smiles = smile_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
    #draw on detected faces and features
    draw_found_faces(detected_faces, frame, (randrange(128),randrange(128),randrange(128)))
    draw_found_faces(detected_eyes, frame, (randrange(128),randrange(128),randrange(128)))
    #draw_found_faces(detected_smiles, frame, (randrange(128),randrange(128),randrange(128)))

    # Display the updated frame as a video stream
    cv2.imshow('Austin Face Detection', frame)

    # Press the ESC key to exit the loop
    # 27 is the code for the ESC key
    if cv2.waitKey(1) == 113 or cv2.waitKey(1)==81:
        break

# Releasing the webcam resource
video_capture.release()

# Destroy the window that was showing the video stream
cv2.destroyAllWindows()'''
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
# iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img = cam.read()
    img = cv2.flip(img, -1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
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

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
