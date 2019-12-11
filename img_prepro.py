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
                        127,255,
                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel,2)
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel,2)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

#for val in centroids:
#cv2.putText(img,"*",(centroids[:,0].astype(int),centroids[:,1].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
centroids
cv2.imshow('Img', img)




