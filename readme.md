## Problem Statement:
    Using Computer Vision techniques, Count the CARS and BIKES moving in each direction of the lane

### Approch:
1. Read copy of Video, convert to gray-scale/Binarise. [**Image pre-processing**]
2. Understand Background and Foreground. [**Image Segmentation**]
3. Perform or detect CAR & BIKE in an Img frame. [**Object Detection**]
4. Perform tracking of the CAR & BIKE [**Object Tracking**] 


#### Refer:
1. https://docs.opencv.org/4.1.0/d2/d55/group__bgsegm.html
2. https://www.geeksforgeeks.org/background-subtraction-in-an-image-using-concept-of-running-average/
3. **https://docs.opencv.org/4.1.0/de/dca/classcv_1_1bgsegm_1_1BackgroundSubtractorCNT.html#details**