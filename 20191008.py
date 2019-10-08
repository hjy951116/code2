import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import csv
import glob
import math
from PIL import Image
import sys


with open('./test.csv','r') as csvfile:
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

P = []
Pl = 0
Ph = 0
F = numpy.zeros((720,1280))
F1 = numpy.zeros((720,1280))
# image = cv2.imread('000000.jpg')
# Ec = image
# Ec1 = image
# Ec2 = image
Ec = numpy.zeros((720,1280,3))
Ec1 = numpy.zeros((720,1280,3))
Ec2 = numpy.zeros((720,1280,3))

count = 0

# Open frames in the folder
for frames in glob.glob('./test/*.jpg'):
  img = cv2.imread(frames)
  ycbcr = Image.open(frames)
  y, cb, cr = ycbcr.split()
  y.show()
  Y = numpy.array(y)
#   print(Y)
  gray_levels = 256
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  hist = numpy.histogram(Y,bins=gray_levels)
  cv2.imwrite('%.1d-gray.jpg'%count,gray)
  
  # print(hist)
  im = Image.open(frames)
  pix = im.load()
  cv2.imwrite('%.1d.jpg'%count,img)
#   r = img[...,2]
#   g = img[...,1]
#   b = img[...,0]
  width = im.size[0]
  height = im.size[1]
  
  dst = cv2.bilateralFilter(Y, d=10, sigmaColor=100, sigmaSpace=100)
  cv2.imwrite('%.6d.jpg'%count,dst)
  Ym = numpy.mean(Y)
  Yrange = numpy.max(Y)-numpy.min(Y)
  YA = 0.5 + Ym - dst
#   print(Ym)
#   print(YA)

  if column[k] == 1:
    for j in range (255):
      f = hist[0]
      P.append(float(f[j])/float(921600))
      w = abs(j-127)/255 + 0.7
      if j < 255-Ym :
        Pl += w*float(f[j])/float(921600)
      else:
        Ph += w*float(f[j])/float(921600)
    for m in range (720):
      for n in range (1280):
        if Y[m][n] < Ym:
          F[m][n] = YA[m][n]*(sum(P[:int(255-Y[m][n])]))/Pl
          F1[m][n]=255-F[m][n]
          for c in range (3):
            Ec1[m][n][c] = img[m,n,c]*F1[m][n]/Y[m][n]
            Ec2[m][n][c] = (Ec1[m][n][c]+img[m,n,c]+F1[m][n]-Y[m][n])/2
            Ec[m][n][c] = Ec2[m][n][c] + (img[m,n,c]-Y[m][n])*(Ym/Yrange)
        else:
        # print(YA[x][y])
        # print(sum(P))
        # print(sum(P[:int(255-gray[x][y])]))
        # print(Ph)
          F[m][n] = YA[m][n]+(1-YA[m][n])*(sum(P)-sum(P[:int(255-Y[m][n])]))/Ph
          F1[m][n]=255-F[m][n]
          for c in range (3):
            Ec1[m][n][c] = img[m,n,c]*F1[m][n]/Y[m][n]
            Ec2[m][n][c] = (Ec1[m][n][c]+img[m,n,c]+F1[m][n]-Y[m][n])/2
            Ec[m][n][c] = Ec2[m][n][c] + (img[m,n,c]-Y[m][n])*(Ym/Yrange)
    cv2.imwrite('%.6d-comp.jpg'%count,F1)
    print(Ec)
    cv2.imwrite('%.6d-const.jpg'%count,Ec)
    

     
    # plt.imshow(adjustedpixel, interpolation='nearest')
    # plt.show()
    # im.show()


  count += 1
  k += 1