import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.mp4')
#fgbgKNN = cv2.createBackgroundSubtractorKNN()
fgbgMOG = cv2.bgsegm.createBackgroundSubtractorMOG(100,5,0.7,0)
#fgbgGMG = cv2.bgsegm.createBackgroundSubtractorGMG()
#fgbgMOG2 = cv2.createBackgroundSubtractorMOG2()
#fgbgCNT = cv2.bgsegm.createBackgroundSubtractorCNT(15,True,15*60,True)

while(1):
    ret, frame = cap.read()
#   fgmaskKNN = fgbgKNN.apply(frame)
    fgmaskMOG = fgbgMOG.apply(frame)
#   fgmaskGMG = fgbgGMG.apply(frame)
#   fgmaskMOG2 = fgbgMOG2.apply(frame)
#   fgmaskCNT = fgbgCNT.apply(frame)
#   
#   cv2.imshow('frame',frame)
#   cv2.imshow('fgmaskKNN',fgmaskKNN)
    cv2.imshow('fgmaskMOG',fgmaskMOG)
#   cv2.imshow('fgmaskGMG',fgmaskGMG)
#   cv2.imshow('fgmaskMOG2',fgmaskMOG2)
#   cv2.imshow('fgmaskCNT',fgmaskCNT)

    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()