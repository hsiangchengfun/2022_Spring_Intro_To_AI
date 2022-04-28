import os
import cv2
import numpy as np

def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    
    subfolders = os.listdir(dataPath)
    for subfolder in subfolders:
      subPath = dataPath + '/'+ subfolder
      iptfiles = os.listdir(subPath)
      
      for iptfile in iptfiles:
        imgPath = subPath + '/' + iptfile
        img = cv2.imread(imgPath)
        
        if(subfolder == 'car'):
          label = 1 
        elif(subfolder == 'non-car'):
          label = 0 
        
        img_1 = cv2.resize(img,(36,16),interpolation = cv2.INTER_AREA)
        
        img_2 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        tuple = (img_2, label)  
        dataset.append(tuple)
    
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    
    return dataset
