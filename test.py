import numpy as np
import cv2
import matplotlib.pyplot as plt


path_to_video = 'sample2.mp4'
cap = cv2.VideoCapture(path_to_video)

fgbg = cv2.createBackgroundSubtractorMOG()



while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        fgmask = fgbg.apply(frame)
        plt.imshow(fgmask)
    else:
        break
cap.release()