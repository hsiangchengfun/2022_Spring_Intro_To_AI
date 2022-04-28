import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
ret,frame = cap.read()
frame=cv2.resize(frame,(0,0),fx=0.15,fy=0.15)
while(1):
    
    ret2,frame2 = cap.read()
    if(ret2):
        
        
        
        frame2=cv2.resize(frame2,(0,0),fx=0.15,fy=0.15)
        
        result = cv2.absdiff(frame,frame2)
        b,g,r=cv2.split(result)
        zeros = np.zeros(result.shape[:2],dtype = "uint8")
        merged_g=cv2.merge([zeros,g,zeros])
        
        
        
        
        comb = np.hstack((frame,merged_g))
        cv2.imshow("img",comb)
        
        
        frame=frame2.copy()
    else:break
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows
    
        