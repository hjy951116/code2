import cv2 
import glob
from PIL import Image
for frames in glob.glob('./test/*.jpg'):
  img = cv2.imread(frames)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
 
  cv2.imshow('image', img[...,2]) 
  cv2.waitKey(0)		 
  cv2.destroyAllWindows()
    