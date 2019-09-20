import numpy as numpy
import cv2
import os
import glob
from PIL import Image
for frames in glob.glob('./7/*.jpg'):
  img = Image.open(frames)
  pixels = img.load() # create the pixel map

  for i in range(img.size[0]): # for every pixel:
    for j in range(img.size[1]):
      pixels[i,j] = (0, 0 ,100)

img.show()