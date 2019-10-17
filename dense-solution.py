import cv2 as cv
import numpy as np
import csv
import glob
import math
from PIL import Image

with open('./test.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))
k = 0
count = 0
previous_img = cv.imread('./test/000000.jpg')
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("test1.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    cv.imshow("input", frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    for frames in glob.glob('./test/*.jpg'):
      img = cv.imread(frames)
      imgcomp = img
#   gray_levels = 256
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
      yuvimg = yuv[...,0]
      U = yuv[...,1]  #0.492 * (img[...,0] - gray)
      V = yuv[...,2]  #0.877 * (img[...,2] - gray)
      im = Image.open(frames)
      pix = im.load()
      width = im.size[0]
      height = im.size[1]
      if column[k] == 1:
        for m in range (720):
          for n in range (1280):
              if int(m-flow[m,n, 1]) < 720 and int(m-flow[m,n, 1]) >= 0 and int(n-flow[m,n, 0]) < 1280 and int(n-flow[m,n, 0]) >= 0:
                imgcomp[m,n,2] =  previous_img[int(m-flow[m,n, 1]),int(n-flow[m,n, 0]),2]
                imgcomp[m,n,1] =  previous_img[int(m-flow[m,n, 1]),int(n-flow[m,n, 0]),1]
                imgcomp[m,n,0] =  previous_img[int(m-flow[m,n, 1]),int(n-flow[m,n, 0]),0]
        print(k)
        cv.imwrite('%.1d-comp.jpg'%count,img)
      count += 1
      k += 1

      previous_img = img
      
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # print(magnitude, angle)
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray

    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()

