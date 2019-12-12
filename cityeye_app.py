# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:26:37 2019
@author: apotdar

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import time

path_to_video = 'vids/trim_road_traffic.mp4'
cap = cv2.VideoCapture(path_to_video)


#===============================================================================
#Info about Video file
#===============================================================================
print("Frame Count in Video: ",cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame Height: ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame Width: ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))



#===============================================================================
def applyPreprocessing(frame,bgsegm_inst,kern_len):
    """
    """    
    proc_frame = cv2.GaussianBlur(frame, (5, 5), 0)    
    #Apply background degmentation algo.
    proc_frame = bgsegm_inst.apply(proc_frame)    
    # Apply Morphological ops tO get clean and noise free blob/object detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kern_len,kern_len))
    proc_frame = cv2.morphologyEx(proc_frame, cv2.MORPH_CLOSE, kernel)
    proc_frame = cv2.morphologyEx(proc_frame, cv2.MORPH_OPEN, kernel)
    proc_frame = cv2.morphologyEx(proc_frame, cv2.MORPH_DILATE, kernel)    
    return proc_frame

def findObjectProps(proc_frame):
    """
    """
    # Extract connected objects and its Centroid[x,y], stats[BBox(0:3),Area(4)]
    _, _, stats, centroids = cv2.connectedComponentsWithStats(proc_frame)        
    centroids = np.reshape(centroids[~np.isnan(centroids)],(-1,2))
    return stats, centroids.astype(np.uint16)

def detectVehicles(og_frame,stats,centroids,disp_area=None,disp_center=None):
    left_lane_cnt = 0
    right_lane_cnt = 0
    #Now that we have all the detected components/objects and its stats in a frame we will
    #iterate through each of the detected object and apply rule to get targeted
    for idx,obj_xy in enumerate(centroids):
        
        #Hypothetical line boundry, count vehicles upon crossing 
        cv2.line(frame,(120,240),(280,240),(0,255,255),2)
        cv2.line(frame,(340,200),(450,200),(0,255,128),2)
        
        #try to filter out actual targeted objects(CARS) based on Area, 
        if stats[idx,4] > 400 and stats[idx,4] < 8000 :
            
            # Logic to count the vehicles passing through line boundry
            if frame[int(obj_xy[1]),int(obj_xy[0]),1] == 255 and frame[int(obj_xy[1]),int(obj_xy[0]),2] == 255:
                left_lane_cnt = left_lane_cnt+1                    
            if frame[int(obj_xy[1]),int(obj_xy[0]),1] == 255 and frame[int(obj_xy[1]),int(obj_xy[0]),2] == 128:
                right_lane_cnt = right_lane_cnt+1
                
            #======================Put Info on Frame=======================
            
            # Centroid Vals of the detected objects in a frame
#            centr_str = str(obj_xy[0].astype(np.uint32))+","+str(obj_xy[1].astype(np.uint32))
#                cv2.putText(frame,centr_str,(int(obj_xy[0]),int(obj_xy[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            
            # Area Vals of the detected objects in a frame
#            area_str = str(stats[idx,4])
#                cv2.putText(frame,area_str,(int(obj_xy[0]),int(obj_xy[1]+20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,240,0))
            
            # Update counter info
            cv2.putText(frame,'Left Lane Count:'+str(left_lane_cnt),(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,255),2)
            cv2.putText(frame,'Right Lane Count:'+str(right_lane_cnt),(470,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,128),2)
    return frame


start_time = time.time()
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
left_lane_cnt = 0
right_lane_cnt = 0

#OutVideo = cv2.VideoWriter('cityeye_action.mp4',cv2.VideoWriter_fourcc(*'H264'),30,(640,360))
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        #==================Perform preprocessing==============================
        # Apply smooth filter to reduce any noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        #Apply background degmentation algo.
        fgmask = fgbg.apply(frame)
        
        # Apply Morphological ops tO get clean and noise free blob/object detection
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
        
#        # Extract connected objects and its Centroid[x,y], stats[BBox(0:3),Area(4)]
#        nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(fgmask)        
#        centroids = np.reshape(centroids[~np.isnan(centroids)],(-1,2))
#        
#        #Now that we have all the detected components/objects and its stats in a frame we will
#        #iterate through each of the detected object and apply rule to get targeted
#        for idx,obj_xy in enumerate(centroids):
#            
#            #Hypothetical line boundry, count vehicles upon crossing 
#            cv2.line(frame,(120,240),(280,240),(0,255,255),2)
#            cv2.line(frame,(340,200),(450,200),(0,255,128),2)
#            
#            #try to filter out actual targeted objects(CARS) based on Area, 
#            if stats[idx,4] > 400 and stats[idx,4] < 8000 :
#                
#                # Logic to count the vehicles passing through line boundry
#                if frame[int(obj_xy[1]),int(obj_xy[0]),1] == 255 and frame[int(obj_xy[1]),int(obj_xy[0]),2] == 255:
#                    left_lane_cnt = left_lane_cnt+1                    
#                if frame[int(obj_xy[1]),int(obj_xy[0]),1] == 255 and frame[int(obj_xy[1]),int(obj_xy[0]),2] == 128:
#                    right_lane_cnt = right_lane_cnt+1
#                    
#                #======================Put Info on Frame=======================
#                
#                # Centroid Vals of the detected objects in a frame
#                centr_str = str(obj_xy[0].astype(np.uint32))+","+str(obj_xy[1].astype(np.uint32))
##                cv2.putText(frame,centr_str,(int(obj_xy[0]),int(obj_xy[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
#                
#                # Area Vals of the detected objects in a frame
#                area_str = str(stats[idx,4])
##                cv2.putText(frame,area_str,(int(obj_xy[0]),int(obj_xy[1]+20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,240,0))
#                
#                # Update counter info
#                cv2.putText(frame,'Left Lane Count:'+str(left_lane_cnt),(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(128,0,255),2)
#                cv2.putText(frame,'Right Lane Count:'+str(right_lane_cnt),(470,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,128),2)
#
#                
##        OutVideo.write(frame)
##        cv2.imshow('Video', fgmask)
        cv2.imshow('ProVideo', fgmask)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:        
        break
    
cap.release()
#OutVideo.release()
cv2.destroyAllWindows()


print("Prog Execution Time= %s Seconds"%(time.time()-start_time))
