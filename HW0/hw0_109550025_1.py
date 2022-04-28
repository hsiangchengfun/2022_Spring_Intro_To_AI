import cv2

fname="image.png"
img=cv2.imread(fname)


list1 = []
list1 = open('bounding_box.txt', "r").readlines()
for i in list1:
    a=i.split()
    cv2.rectangle(img,(int(a[0]) ,int(a[1])) ,(int(a[2]) ,int(a[3])),(0,0,255), 4)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.imwrite('hw0_109550025_1.jpg',img)

 