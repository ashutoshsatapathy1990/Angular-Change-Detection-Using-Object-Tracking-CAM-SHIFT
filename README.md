# ANGULAR CHANGE DETECTION USING OBJECT TRACKING (CAM SHIFT)
#Object tracking using CAM shift and histogram back projection algorithm
Histogram back projection algorithm is sensitive to noise and continous change in color. It works effectively on objects having uniform foreground and backgound. It computes histogram of selected ROI on current frame which will be back project on next frame to calculate ROI.  
#
CAM shift algorithm tracks points of Region of interest where size of the bounded box varies dynamically.
#
#About the program
Actual ROI is selected manually by pressing keyboard key "i". It allows to select four corner points for tracking purposes. These four points are used to generate bounding box by camshift algorithm  dynamically. Each subsequent iteration, the newly generated four points are used to calculate centroid of actual ROI.
#
Another ROI is selected manually by pressing keyboard key "s". It allows to select four corner points for tracking purposes, but these points are used to calculate reference point where reference point is the centroid of current ROI.
#
Slope, angle and angle difference between successive frames are calculated using centroid of actual ROI and reference point.
