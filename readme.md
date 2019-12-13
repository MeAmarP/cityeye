## Problem Statement:
    Using Computer Vision techniques, Count the CARS/BIKES moving in each direction of the lane

### Approch:
1. Capture Video data as frames
2. Understand Background and Foreground. [**Image Segmentation**]
3. Perform necessary filtering and morphing to get better blob/object detection. 
4. Perform or detect CAR/BIKE in an Img frame. [**Object Detection**]
5. Apply rule based filter(Area/length) to detect/track ONLY vehicles in a given frame.
6. Perform tracking of the object, check if crosses defined hypothetical border to keep count. [**Object Tracking**]
7. Use shape stats(Height to width ratio) extracted from contours To classify between TRUCKS, CARS and BIKE.




#### Refer:
1. https://docs.opencv.org/4.1.0/d2/d55/group__bgsegm.html
2. **https://docs.opencv.org/4.1.0/de/dca/classcv_1_1bgsegm_1_1BackgroundSubtractorCNT.html#details**
