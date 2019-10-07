# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread('000000.jpg')
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()

import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy import misc
import scipy.misc

img = scipy.misc.imread('000003.jpg')


lu1=img[...,2].flatten()
plt.subplot(3,1,1)
plt.hist(lu1,bins=256,range=(0.0,255.0),histtype='stepfilled', color='r', label='Red')
# plt.title("Red")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

lu2=img[...,1].flatten()
plt.subplot(3,1,2)                  
plt.hist(lu2,bins=256,range=(0.0,255.0),histtype='stepfilled', color='g', label='Green')
# plt.title("Green")   
plt.xlabel("Value")    
plt.ylabel("Frequency")
plt.legend()

lu3=img[...,0].flatten()
plt.subplot(3,1,3)                  
plt.hist(lu3*255,bins=256,range=(0.0,255.0),histtype='stepfilled', color='b', label='Blue')
# plt.title("Blue")   
plt.xlabel("Value")    
plt.ylabel("Frequency")
plt.legend()
plt.show()