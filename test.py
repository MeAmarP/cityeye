# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:26:37 2019
@author: apotdar

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import time

path_to_video = 'F:/Dataset/forFatNinja/cityeye/trim_HwayTraffic.mp4'
cap = cv2.VideoCapture(path_to_video)


#===============================================================================
#Info about Video file
#===============================================================================
print("Frame Count in Video: ",cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame Height: ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame Width: ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))



start_time = time.time()
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.medianBlur(frame,5)
        fgmask = fgbg.apply(frame)        
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#        cv2.imshow('ProcessedVideo', fgmask)
        cv2.imshow('Video', cv2.hconcat([frame,fgmask]))
        
#        cv2.imshow('OriginalVideo', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


print("Prog Execution Time= %s Seconds"%(time.time()-start_time))
