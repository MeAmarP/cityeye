# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:26:37 2019
@author: apotdar

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import time

path_to_video = 'vids/Hway_traffic.mp4'
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
left_lane_cnt = 0
right_lane_cnt = 0
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
        for idx,obj_xy in enumerate(centroids):
            cv2.line(frame,(120,240),(280,240),(0,255,255),2)
            cv2.line(frame,(345,210),(450,210),(0,255,128),2)
            if stats[idx,4] > 400 and stats[idx,4] < 8000 :
                if frame[int(obj_xy[1]),int(obj_xy[0]),1] == 255 and frame[int(obj_xy[1]),int(obj_xy[0]),2] == 255:
                    left_lane_cnt = left_lane_cnt+1                    
                if frame[int(obj_xy[1]),int(obj_xy[0]),1] == 255 and frame[int(obj_xy[1]),int(obj_xy[0]),2] == 128:
                    right_lane_cnt = right_lane_cnt+1                    
                centr_str = str(obj_xy[0].astype(np.uint32))+","+str(obj_xy[1].astype(np.uint32))
                area_str = str(stats[idx,4])
                cv2.putText(frame,'Left Lane Count:'+str(left_lane_cnt),(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,255),2)
                cv2.putText(frame,'Right Lane Count:'+str(right_lane_cnt),(470,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,128),2)
#                cv2.putText(frame,area_str,(int(obj_xy[0]),int(obj_xy[1]+20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,240,0))
#                cv2.putText(frame,centr_str,(int(obj_xy[0]),int(obj_xy[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.imshow('Video', frame)
#        print('Left Lane Count',left_lane_cnt)
#        print('Right Lane Count',right_lane_cnt)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


print("Prog Execution Time= %s Seconds"%(time.time()-start_time))
