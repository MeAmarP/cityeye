# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:26:37 2019
@author: apotdar

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import time

path_to_video = 'vids/trim_Hway_traffic.mp4'
cap = cv2.VideoCapture(path_to_video)


#===============================================================================
#Info about Video file
#===============================================================================
print("Frame Count in Video: ",cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame Height: ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame Width: ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))



start_time = time.time()
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        fgmask = fgbg.apply(frame)        
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
        nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
        centroids = np.reshape(centroids[~np.isnan(centroids)],(-1,2))
        for obj_xy in centroids:
            cv2.putText(frame,'*',(int(obj_xy[0]),int(obj_xy[1]) ),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.imshow('Video', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


print("Prog Execution Time= %s Seconds"%(time.time()-start_time))
