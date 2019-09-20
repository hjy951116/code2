import numpy as numpy
import cv2
import os
import matplotlib.pyplot as plt
import math
import csv

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
video_path=r"./vtest.mp4"
 

img_path =r'./images'
img_path2 =r'./equimages'

# if not os.path.isdir(img_path):
#    mkdir(img_path)

# # Divide the video into frames
# vidcap = cv2.VideoCapture(video_path)
# (cap,frame)= vidcap.read()
 
# if cap==False:
#     print('cannot open video file')
# count = 0

# # Save the frames in a folder
# while cap:
#   cv2.imwrite(os.path.join(img_path,'%.6d.jpg'%count),frame)

#   count += 1
#   # Every 100 frames get 1
#   for i in range(1):
#     (cap,frame)= vidcap.read()

framer = []
frameg = []
frameb = []
framera = []
framerb = []
framerc = []
deltay = []
deltaya = []
deltayb = []
deltayc = []
deltab = []
deltab2 = []
i = 0
j = 0
f = 0
frameindex = []
xa = []
xb = []
xc = []
r = []
g = []
b = []
# previous_gray = cv2.cvtColor(cv2.imread('./images/000000.jpg'), cv2.COLOR_BGR2GRAY)
previous_y = numpy.mean(cv2.cvtColor(cv2.imread('./images/000000.jpg'), cv2.COLOR_BGR2GRAY))
previous_blocky = numpy.zeros(1947)
previous_blocky2 = numpy.zeros(1947)
# Open frames in the folder
import glob
from PIL import Image
for frames in glob.glob('./1/*.jpg'):
  img = cv2.imread(frames)
  gray_levels = 256
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  im = Image.open(frames)
  pix = im.load()
  width = im.size[0]
  height = im.size[1]

  # Define the window size
  windowsize_r = 32
  windowsize_c = 32
  
#   m = int(height/16)
#   n = int(width/16)
#  The average luminance component (Y) of an entire frame
  for m in range(width):
      for n in range(height):
        R, G, B = pix[m, n]

        
  rmean = numpy.mean(R)
  #print(rmean)
  gmean = numpy.mean(G)
  bmean = numpy.mean(B)
  framer.append(rmean)
  print(framer)
  frameg.append(gmean)
  frameb.append(bmean)
#  biii = y1-previous_y
#   if biii >= 3:
#       print(j)
#   j += 1
  # bmax = max(y1, previous_y)
  # bi = numpy.power(y1-previous_y,2)
  # bii = numpy.sum(bi)
  # biii = math.sqrt(bii)
  # deltay.append(bii)
  # previous_y = y1


  if column[i] == 1:
    xa.append(i)
    framera.append(framer[i])
    #deltaya.append(deltay[i])
  elif column1[i] == 1:    
    xb.append(i)
    framerb.append(framer[i])
    #deltayb.append(deltay[i])
  else:
    xc.append(i)
    framerc.append(framer[i])
    #deltayc.append(deltay[i])
  i = i+1
print(xa)
n = len(deltay)
x = range(0,n)
# for j in range (1,n-2):
    
#   Hn0 = abs(deltay[j]- deltay[j-1])
#   Hn = abs(deltay[j+1]- deltay[j])
#   Hn1 = abs(deltay[j+2]- deltay[j+1])
#   Hmax = max(Hn, Hn1)
#   #print(Hn)
#   if deltay[j]>deltay[j-1] and deltay[j+1]>deltay[j+2] and deltay[j+1]>deltay[j]:
#   #if deltay[j]>20:
#     if Hn0 > 30 and Hn1 > 30:
#       print(j)
    

plt.axhline(y = 20, color = 'g', linestyle = '-')
plt.axhline(y = 300, color = 'r', linestyle = '-')
plt.axhline(y = 0, linestyle = '-')
plt.scatter(xa,framera,c='red')
plt.scatter(xb,framerb, marker = '^',c='green')
plt.scatter(xc,framerc)
plt.show()