import cv2
import glob
from PIL import Image
for frames in glob.glob('./1/*.jpg'):
  img = cv2.imread(frames)
  im = Image.open(frames)
  pix = im.load()
  width = im.size[0]
  height = im.size[1]

  for m in range(width):
    for n in range(height):
      r, g, b = pix[m, n]