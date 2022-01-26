import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('Photos/sidewalk.jpeg')
img = rescaleFrame(img, 0.75)
#cv.imshow('OG', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)
cvtColor(canny, COLOR_GRAY2BGR )
cv.imshow('Canny', canny)

ret, thresh = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
#cv.imshow('Thresh', thresh)

contours, heirarchies = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

blank = np.zeros(img.shape, dtype='uint8')
cv.drawContours(img, contours, -1, (0,0,255), 1)
#cv.imshow('Contours Drawn', img)

print(len(contours))


cv.waitKey(0)