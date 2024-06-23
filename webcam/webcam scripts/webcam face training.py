# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:36:53 2021

@author: HP
"""
import cv2
import numpy as np
from PIL import Image
import os


# Path for face image database
count=1
path=np.array(os.listdir('../webcam face dataset/'))
for dirpath in path:
    print(str(dirpath))
    createTrainerDir = '../trainer'
    try:    
        os.makedirs(createTrainerDir)
        createTrainer = open('../trainer/trainer.yml','x')
        print('directory',createTrainerDir,'created')
    except FileExistsError:
        print('exists')
    if not os.path.exists(createTrainerDir):
        os.mkdir(createTrainerDir)
    else:
        print('existing')
    dirpath = '../webcam face dataset/' + str(dirpath) 
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("../cascades/haarcascade_frontalface_default.xml");
    
    # function to get the images and label data
    def getImagesAndLabels(dirpath):
    
        imagePaths = [os.path.join(dirpath,f) for f in os.listdir(dirpath)]     
        faceSamples=[]
        ids = []
    
        for imagePath in imagePaths:
    
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
    
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
    
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
    
        return faceSamples,ids
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(dirpath)
    recognizer.train(faces, np.array(ids))
    
    # Save the model into trainer/trainer.yml
    recognizer.write('../trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
    count+=1
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    print('code completed')