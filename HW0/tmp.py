import cv2 

cap = cv2.VideoCapture('video.mp4')
j=0;
while(1):
    ret,frame=cap.read()
    if(j == 0):
        bg=frame.copy()
    if(j!=0):
        cv2.accumulateWeighted(frame,bg,0.5)
        j=j+1
        
    
    diff = cv2.absdiff(frame,bg)
    # diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    
    
    thre,diff=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)
    diff=cv2.resize(diff,(0,0),fx=0.25,fy=0.25)
    cv2.imshow("j",diff)
    if(cv2.waitKey(1) & 0XFF == ord('q')):
        break
    bg=frame.copy()
    
    
cv2.waitKey(0)
        
cap.release()
cv2.destroyAllWindows

