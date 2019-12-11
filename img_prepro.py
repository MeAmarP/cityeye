# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:26:37 2019
@author: apotdar

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

path_to_img = 'vlcsnap_hwy_traffic.png'

img = cv2.imread(path_to_img)
main_img = cv2.GaussianBlur(img, (5, 5), 0)

main_img = cv2.cvtColor(main_img,cv2.COLOR_BGR2GRAY)
_,bin_img = cv2.threshold(main_img,
                        80,255,
                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)



kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

for idx,val in enumerate(centroids):   
    cv2.putText(img,
                str(stats[idx,4]),
                (int(val[0]),int(val[1])),
                cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0))
cv2.imshow('Img', bin_img)




