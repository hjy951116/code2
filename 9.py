import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd
from scipy.signal import find_peaks

with open('./test.csv','rb') as csvfile:
  reader0 = csv.reader(csvfile)
  column0 = [row[1] for row in reader0]
  column0.pop(0)
  column0 = list(map(int,column0))

with open('./test2.csv','rb') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

with open('./test3.csv','rb') as csvfile1:
  reader1 = csv.reader(csvfile1)
  column1 = [row[1] for row in reader1]
  column1.pop(0)
  column1 = list(map(int,column1))

# Open a vidoe


framey = []
frameya = []
frameyb = []
frameyc = []

deltay = []
deltaya = []
deltayb = []
deltayc = []

deltab = []
deltab2 = []

i = 0
j = 0

frameindex = []
frameindex1 = []
flashindex = []
xa = []
xb = []
xc = []
# previous_gray = cv2.cvtColor(cv2.imread('./images/000000.jpg'), cv2.COLOR_BGR2GRAY)
previous_y = numpy.mean(cv2.cvtColor(cv2.imread('./images/000000.jpg'), cv2.COLOR_BGR2GRAY))

# Open frames in the folder
import glob
from PIL import Image
for frames in glob.glob('./6/*.jpg'):
  img = cv2.imread(frames)
  gray_levels = 256
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  im = Image.open(frames)
  pix = im.load()
  width = im.size[0]
  height = im.size[1]
  for m in range(width):
      for n in range(height):
        R, G, B = pix[m, n]
        Y =   16 +  65.738*R/256 + 129.057*G/256 +  25.064*B/256
  y1 = numpy.mean(Y)
  print(y1)
  framey.append(y1)
  bmax = max(y1, previous_y)
  bi = numpy.power(y1-previous_y,2)
  bii = numpy.sum(bi)
  biii = math.sqrt(bi)
  biiii = abs(y1-previous_y)
  deltay.append(biiii)
  previous_y = y1


  if column[i+500] == 1:
    xa.append(i+500)
    deltaya.append(deltay[i])
    frameya.append(framey[i])
  elif column1[i+500] == 1:    
    xb.append(i+500)
    deltayb.append(deltay[i])
    frameyb.append(framey[i])
  else:
    xc.append(i+500)
    deltayc.append(deltay[i])
    frameyc.append(framey[i])
  frameindex1.append(i+500)
  if column0[i+500] == 1:
    flashindex.append(i+500)
  i = i+1
print(flashindex)
n = len(deltay)
x = range(0,n)

# peaks, _ = find_peaks(framey)
# print(peaks) 
   

plt.scatter(xa,frameya,c='red')
plt.scatter(xb,frameyb, marker = '^',c='green')
plt.scatter(xc,frameyc)
plt.plot(frameindex1,framey,color = 'orange')
plt.ylabel('Mean Y Values of Frame')
plt.xlabel('Frame Index')
plt.show()

# plt.scatter(xa,deltaya,c='red')
# plt.scatter(xb,deltayb, marker = '^',c='green')
# plt.scatter(xc,deltayc)
# plt.plot(frameindex1,deltay,color = 'orange')
# plt.ylabel('Mean Y Values Difference of Frames')
# plt.xlabel('Frame Index')
# plt.show()

# d1 = numpy.r_[0, numpy.abs(numpy.array(framey[1:]) - numpy.array(framey[:-1]))]
# d2 = numpy.r_[numpy.abs(numpy.array(framey[1:]) - numpy.array(framey[:-1])), 0]
# print(d1,d2)
# Mask the inliers with a threshold (they have at least another node close):
# frameyarr = numpy.array(framey)
# u = numpy.arange(n)
# mask = (d1 > 1) & (d2 > 1.5)
# print(u[mask])
# plt.plot(u[mask], frameyarr[mask], 'o')
# plt.show()