import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import csv
import glob
import math
from PIL import Image


with open('./test.csv','rb') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))

count2 = 0
framey = []
i = 0
j = 0
k = 0
a = 0.2

r = numpy.zeros((1280, 720))
g = numpy.zeros((1280, 720))
b = numpy.zeros((1280, 720))
adjustedpixel = [[(0,0,0),()]] #numpy.zeros((720,1280,3))
list(zip(a,b,c))
adjustedpixelr = numpy.zeros((720,1280))
adjustedpixelg = numpy.zeros((720,1280))
adjustedpixelb = numpy.zeros((720,1280))
alpha = numpy.zeros((720,1280))
gammaset = numpy.ones((720,1280))

# Open frames in the folder

for frames in glob.glob('./test/*.jpg'):
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
        r[m][n] = R
        g[m][n] = G
        b[m][n] = B
  rm = 128



  if column[k] == 1:
    for x in range (720):
      for y in range (1280): 
        alpha[x][y] = math.pi*r[y][x]/(2*rm)
        gammaset[x][y] = 1 + a*math.cos(alpha[x][y])
        gamma = gammaset[x][y]
        invGamma = 1.0 / gamma
        adjustedpixelr[x][y] = 255 #numpy.array((r[y][x] / 255.0) ** invGamma) * 255
        adjustedpixelg[x][y] = 123 #numpy.array((g[y][x] / 255.0) ** invGamma) * 255
        adjustedpixelb[x][y] = 20 #numpy.array((b[y][x] / 255.0) ** invGamma) * 255
        adjustedpixel[x][y] = adjustedpixelr[x][y],adjustedpixelg[x][y],adjustedpixelb[x][y]
        # adjustedpixel[x][y][1] = adjustedpixelg[x][y]
        # adjustedpixel[x][y][2] = adjustedpixelb[x][y]
    print(k,'flash')
    print(adjustedpixel)
    plt.imshow(adjustedpixel, interpolation='nearest')
    plt.show()

  k += 1


