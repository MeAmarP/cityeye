# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:26:37 2019
@author: apotdar

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

path_to_img = 'img_sample.png'

main_img = cv2.imread(path_to_img)
main_img = cv2.medianBlur(main_img,5)
#img_fltr = cv2.GaussianBlur(main_img,(5,5),0)

main_img = cv2.cvtColor(img_fltr,cv2.COLOR_BGR2GRAY)
_,bin_img = cv2.threshold(main_img,
                        125,255,
                        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)



kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
morph_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

cv2.imshow('Img', morph_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



