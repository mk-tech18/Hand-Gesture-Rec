"""
    This file loads the model and creates a cv window to show the prediction results.
"""
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_SIZE = (160, 160)
detector = tf.keras.models.load_model(os.getcwd()+'lab_inception_v2_resnet_1')

cap = cv2.VideoCapture(-1)
classes = ['come','left','right','stop']

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, IMG_SIZE)
    # We keep consistency as it was the filter we used in data preprocessing
    binary = cv2.inRange(frame,(82,104,134),(132,167,195))
    binary = cv2.Canny(binary, 100, 200)
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    feed = tf.expand_dims(binary,0)
    output = detector.predict(feed)
    res = classes[np.argmax(output)]
    frame = cv2.putText(frame, res,(60,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) , 6)
    cv2.namedWindow('frame', 0) 
    cv2.resizeWindow('frame', 600, 600)
    cv2.imshow('frame',frame)
    cv2.imshow('binary', binary)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()