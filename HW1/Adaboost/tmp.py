
import dataset
import adaboost
import utils
import detection

import os
from turtle import Turtle
import cv2

import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime
from PIL import Image

dataPath = 'data/detect/detectData.txt'


imglist=[]

gifPath = 'data/detect/video.gif'
frame2 = Image.open(r"data/detect/video.gif")

print(type(frame2))
# img = frame2.seek(0)
print(type(frame2.seek(0)))
numframes = 0

img = cv2.cvtColor(np.asarray(frame2),cv2.COLOR_RGB2BGR)
frame2.show()
frame2.save("img.png")
# cv2.imshow("gs",img)
cv2.waitKey(100)
new_img=cv2.imread("img.png")

while frame2:

  try:
    numframes += 1
    print("fff ",numframes)
    frame2.seek(numframes)
    frame = frame2.copy()

    
    
    with open(dataPath, 'r') as f:
      num = int(f.readline())
      for i in range(num):
        data = f.readline()
        list2 = data.split(' ')
        x1,y1,x2,y2,x3,y3,x4,y4 = (int)(list2[0]),(int)(list2[1]),(int)(list2[2]),(int)(list2[3]),(int)(list2[4]),(int)(list2[5]),(int)(list2[6]),(int)(list2[7])

        cvframe = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2GRAY)
        new_cvframe = cv2.resize(cvframe,(360,160),interpolation=cv2.INTER_AREA)
        # print(type(new_cvframe))
        # cv2.imshow("e",new_cvframe)
        # cv2.waitKey(1)
        
        # cv2.rectangle(new_img,(x3,y3),(x2,y2),(0,255,0),1)
        cv2.line(new_img,(x3,y3),(x4,y4),[0,255,0],2)
        cv2.line(new_img,(x3,y3),(x1,y1),[0,255,0],2)
        cv2.line(new_img,(x2,y2),(x1,y1),[0,255,0],2)
        cv2.line(new_img,(x2,y2),(x4,y4),[0,255,0],2)
        # cv2.imshow("w",new_img)
        cv2.waitKey(1)
      list2=[]

      
    imglist.append(new_img)

    
  except EOFError:
    print("etw")
    break

cv2.imshow("t",imglist[0])
cv2.waitKey(0)