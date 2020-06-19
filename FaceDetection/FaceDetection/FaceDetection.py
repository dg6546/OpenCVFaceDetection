import cv2
import os
import numpy as np

#pathj = "C:\\Users\\lion\\source\\repos\\FaceRecognition\\FaceRecognition\\IMG\\9d5c5234a21c70b165476df93ac842e08e089f8cr1-1200-1800v2_uhq.jpg"
pathf = "C:\\Users\\lion\\source\\repos\\FaceRecognition\\FaceRecognition\\cascades\data\\haarcascade_frontalface_alt2.xml"

video_capture = cv2.VideoCapture(0)

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load(pathf)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30))
    return faces

#test_img = cv2.imread(pathj)
#face_detected, gray_img = faceDetection(test_img)

while (video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret:
        faces_detected = faceDetection(frame)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), thickness = 5)
        cv2.imshow('Video', frame)
        cv2.waitKey(25)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#while True:
#    ret, frame = video_capture.read()
#    if ret:
#        cv2.imshow('Video', frame)
#        cv2.waitKey(25)
#    else:
#        print("camera not found")
#        break


video_capture.release()
cv2.destroyAllWindows()







