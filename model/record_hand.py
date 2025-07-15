import cv2
import matplotlib.pyplot as plt
import os

vidcap = cv2.VideoCapture(-1)
count = 0
path = os.getcwd() + '/hand_sign'

def f(x): return x

while vidcap.isOpened():
    ret, frame = vidcap.read()
    cv2.namedWindow('video_window')
    cv2.namedWindow('binary_window')
    #the default color format was GBR
    binary_image = cv2.inRange(frame,(82,104,134),(132,167,195))
    # This shows the original picture from webcam
    cv2.imshow('frame',frame)
    # This shows the filtered binary image
    cv2.imshow('binary_window', binary_image)
    # This is to save data files
    # cv2.imwrite(path+"stop/stop_frame%d.jpg" %count,frame)
    count += 1
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
vidcap.release()
cv2.destroyAllWindows()
