import numpy as np
import cv2 as cv
import sys
import math
import glob
import time

vid = cv.VideoCapture('Photos/SidewalkCampus3_Trim_Clean_Trim.mp4')

def changeRes(width, height):
    vid.set(3, width)
    vid.set(4, height)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0
font = cv.FONT_HERSHEY_SIMPLEX

input_pts = np.float32([[181,498+176],[1258,500+176],[899,293+176],[520,296+176]])
output_pts = np.float32([[512,720],[768,720],[768,0],[512,0]])

M = cv.getPerspectiveTransform(input_pts, output_pts)

while vid.isOpened():
    ret, frame = vid.read()

    #if frame is read correctly ret is True
    if not ret:
        break


    out = cv.warpPerspective(frame, M, (1280,720), flags = cv.INTER_LINEAR)

    # # Color grade conversion for unwarped image
    hsv = cv.cvtColor(out, cv.COLOR_BGR2HSV)

    #cv.imshow('Orignal', frame)
    # Threshold of blue in HSV space
    darkest = np.array([0, 0, 0])
    lightest = np.array([255, 255, 150])
 
    # preparing the mask to overlay
    mask = cv.inRange(hsv, darkest, lightest)
     
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv.bitwise_and(out, out, mask = mask)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)  
    #cv.imshow('Gray', gray)  

    #blur = cv.bilateralFilter(out, 12, 150, 50, cv.BORDER_DEFAULT)
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    #cv.imshow('blur', blur)

    auto = cv.Canny(blur, 75, 175)
    #auto = auto_canny(blur, 1)
    #cv.imshow('Canny',auto)    
    
    #auto = cv.GaussianBlur(auto, (11,11), cv.BORDER_DEFAULT)  
    #cv.imshow('Blurred', auto)
    # sobel = cv.Sobel(blur, cv.CV_64F, 1, 0)
    # out = cv.warpPerspective(sobel, M, (1280,720), flags = cv.INTER_LINEAR)
    #cv.imshow('Canny', canny)

    


    # Calculate HoughLinesTransform
    # Copy edges to the images that will display the results in BGR

    cdst = cv.cvtColor(auto, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    #cdst = np.copy(canny)
    #cdstP = np.copy(canny)

    lines = cv.HoughLines(auto, 1, np.pi / 180, 150, None, 0, 0)

    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            #if b < 0.2 and b > -0.2:
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(out, pt1, pt2, (0,0,255), 2, cv.LINE_AA)
        # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv.putText(out, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)
 
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", out)    
    
    # linesP = cv.HoughLinesP(auto, 1, np.pi / 180, 30, None, minLineLength=30, maxLineGap=150)
    

    
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(out, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", out)    




    #cv.imshow('Warped', canny)

    #cv.imshow('Canny', masked)

    
    #cv.imshow('Bilateral', blur)

    #ret, thresh = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
    #cv.imshow('Thresh', thresh)

    #contours, heirarchies = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if cv.waitKey(1) == ord('q'):
        break

vid.release()
cv.destroyAllWindows()