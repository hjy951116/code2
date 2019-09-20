import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import math
import csv

with open('./test.csv','rb') as csvfile:
  reader = csv.reader(csvfile)
  column = [row[1] for row in reader]
  column.pop(0)
  column = list(map(int,column))


framey = []
i=0

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

  ay = numpy.array(gray).flatten()
  if column[i+500] == 1:
    plt.hist(ay ,bins = 256, density = 1, facecolor = 'red', edgecolor = 'red')
    plt.ylabel('Probability Distribution of Frames with Flash')
    plt.xlabel('Frame Gray Levels')
    plt.show()
  else:
    plt.hist(ay ,bins = 256, density = 1, facecolor = 'blue', edgecolor = 'blue')
    plt.ylabel('Probability Distribution of Frames without Flash')
    plt.xlabel('Frame Gray Levels')
    plt.show()

  i += 1

  values, base = numpy.histogram(gray, bins=256)
#evaluate the cumulative
  cumulative = numpy.cumsum(values)
# plot the cumulative function
  # plt.plot(base[:-1], cumulative, c='green')
  # plt.show()
  print(values)
  y1 = numpy.mean(gray)
  framey.append(y1)
  print(i)