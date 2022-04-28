import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime
from PIL import Image


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)  
    imglist=[]
    ff=open("adaboost_pred.txt","w")
    gifPath = 'data/detect/video.gif'
    
    #from here i read the gif by PIL instead of cv2
    #because according to the source i searched on the internet
    #it refer that cv2 can't read gif(though i foun dsomebody successfully used  it later)
    frame2 = Image.open(r"data/detect/video.gif")
    numframes = 0
    img = cv2.cvtColor(np.asarray(frame2),cv2.COLOR_RGB2BGR)
    #i save the first frame as "img.png"in order to draw bounding_box on it
    frame2.save("img.png")
    
    numframes = 0
    #A While loop to go through all the 50 frames
    while frame2:
      new_img=cv2.imread("img.png")
      try:
          
        numframes += 1
        #frame2.seek(x) means the xth frame of frame2.gif
        frame2.seek(numframes)
        
         
        frame = frame2.copy()
        
        with open(dataPath, 'r') as f:
          #first step we read the number which means the num of parking spaces
          num = int(f.readline())
          #there are 76 plots means we will do th eloop 76 times
          for i in range(num):
            #read the four point pairs 
            data = f.readline()
            #save the points in x1~x4 and y1~y4 with splitting them by a space before
            list2 = data.split(' ')
            x1,y1,x2,y2,x3,y3,x4,y4 = (int)(list2[0]),(int)(list2[1]),(int)(list2[2]),(int)(list2[3]),(int)(list2[4]),(int)(list2[5]),(int)(list2[6]),(int)(list2[7])
            #convert the PIL type img to the cv2 type and convert it to the gray scale from the RGB order
            cvframe = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2GRAY)
            #crop the image by four point pairs by the function that TA gave
            cvframe_2 = crop(x1,y1,x2,y2,x3,y3,x4,y4,cvframe)
            #resize the ing to fit theclasify function
            new_cvframe = cv2.resize(cvframe_2,(36,16),interpolation=cv2.INTER_AREA)
            
            #use classifier to classify each plot image
            #if the return value is 1 
            #then draw the green bounding_box on the border of the plot in the "new_img"
            #which is read from "img.png" we saved before 
            #also write "1 " onto the adaboost_pred.txt
            if(clf.classify(new_cvframe)):
                ff.write('1 ')
                cv2.line(new_img,(x3,y3),(x4,y4),[0,255,0],2)
                cv2.line(new_img,(x3,y3),(x1,y1),[0,255,0],2)
                cv2.line(new_img,(x2,y2),(x1,y1),[0,255,0],2)
                cv2.line(new_img,(x2,y2),(x4,y4),[0,255,0],2)
            #if classifier return 0 ,then write "0" to adaboost_pred.txt
            else:
                ff.write('0 ')
        #refresh the list2 ,and append the new_img to the imglist
        list2=[]  
        imglist.append(new_img)

        ff.write("\n")  
      #if the frame has been the tail of the gif,then break
      except EOFError:
        print("end")
        break     
    #follow  the instruction that the spec said 
    #so i show the first element in the imglist 
    #which is the first frame
    cv2.imshow("0",imglist[0])
    ff.close()
    cv2.waitKey(0)


