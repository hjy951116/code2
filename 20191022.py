import cv2 as cv
import numpy as np
import csv
import math
import pandas as pd


with open('./test2.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))
k = 0
count = 0
pos = np.zeros((720,1280,2))
print(pos)

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def bicubic(fx, fy, img):

   img = img.astype(float)
  #  h,w = np.shape(img)
   h = 720
   w = 1280

   if(fx < 0 or fx > w-1 or fy < 0 or fy > h-1):
       val = 0
       return val

   x = int(fx)
   y = int(fy)

   dx = fx - x
   dy = fy - y

   px = 0 if x-1<0 else x-1
   py = 0 if y-1<0 else y-1

   nx = x+1 if dx>0 else x
   ny = y+1 if dy>0 else y

   ax = w-1 if x+2>=w else x+2
   ay = h-1 if y+2>=h else y+2

   Ipp = img[py,px]; Icp = img[py,x]; Inp = img[py,nx]; Iap = img[py,ax]
   Ip = Icp + 0.5*(dx*(-Ipp+Inp) + dx*dx*(2*Ipp-5*Icp+4*Inp-Iap) + dx*dx*dx*(-Ipp+3*Icp-3*Inp+Iap))
   Ipc = img[y,px]; Icc = img[y,x]; Inc = img[y,nx]; Iac = img[y,ax]
   Ic = Icc + 0.5*(dx*(-Ipc+Inc) + dx*dx*(2*Ipc-5*Icc+4*Inc-Iac) + dx*dx*dx*(-Ipc+3*Icc-3*Inc+Iac))
   Ipn = img[ny,px]; Icn = img[ny,x]; Inn = img[ny,nx]; Ian = img[ny,ax]
   In = Icn + 0.5*(dx*(-Ipn+Inn) + dx*dx*(2*Ipn-5*Icn+4*Inn-Ian) + dx*dx*dx*(-Ipn+3*Icn-3*Inn+Ian))
   Ipa = img[ay,px]; Ica = img[ay,x]; Ina = img[ay,nx]; Iaa = img[ay,ax]
   Ia = Ica + 0.5*(dx*(-Ipa+Ina) + dx*dx*(2*Ipa-5*Ica+4*Ina-Iaa) + dx*dx*dx*(-Ipa+3*Ica-3*Ina+Iaa))

   val = Ic + 0.5*(dy*(-Ip+In) + dy*dy*(2*Ip-5*Ic+4*In-Ia) + dy*dy*dy*(-Ip+3*Ic-3*In+Ia))
   return val

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("test.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
previous_img = frame
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(frame)
# Sets image saturation to maximum
mask[..., 1] = 255

x = np.arange(0,1280)
y = np.arange(0,720)

# while(cap.isOpened()):
for n in range(10):
    previous_img = frame.copy()
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    cv.imshow("input", frame)
    cv.imshow("prev", previous_img)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)      
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # print(magnitude, angle)
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray

    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    img = frame

    imgcomp = img
    a = []
    b = []
    c = []
    d = []
    if column[k+1] == 1:
      for m in range (720):
        for n in range (1280):
          a.append(m-flow[m, n, 1])
          b.append(n-flow[m, n, 0])
          # c.append(magnitude[m, n])
          # d.append(angle[m, n])
          # interpolationflow = bicubic(flow[m, n, 0], flow[m, n, 1], img)
          # print(interpolationflow)
          # if int(m-flow[m, n, 1]) < 720 and int(m-flow[m,n,1]) >= 0 and int(n-flow[m,n,0]) < 1280 and int(n-flow[m,n,0]) >= 0:
          #   imgcomp[m,n,2] =  previous_img[int(m-flow[m,n,1]),int(n-flow[m,n,0]),2]
          #   imgcomp[m,n,1] =  previous_img[int(m-flow[m,n,1]),int(n-flow[m,n,0]),1]
          #   imgcomp[m,n,0] =  previous_img[int(m-flow[m,n,1]),int(n-flow[m,n,0]),0]

      print(k)
      
      raw_data = {'y': a, 
      'x': b}
      # ,
      # 'magnitude': c,
      # 'angle': d}
      df = pd.DataFrame(raw_data , columns = ['y', 'x']) #, 'magnitude', 'angle'])
      df.to_csv('example.csv', index=False)
      cv.imwrite('%.1d-comp.jpg'%count,imgcomp)
    count += 1
    k += 1


    



      

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
