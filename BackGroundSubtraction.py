import numpy as np
import cv2
import os
from os.path import isfile, join
from tifffile import imsave
from os import listdir





# creating object
fgbg2 = cv2.createBackgroundSubtractorMOG2()
lis = listdir("modified_training/Training_row")

for j in range(len(lis)):

# capture frames from a camera

    path ="modified_training/Training_row/" + lis[j] + "/%04d.tif"
    cap = cv2.VideoCapture(path)
    print(cap.isOpened())
    i = 0

    secondPath = "modified_training/Training_modified/" + lis[j]
    lis2 = listdir(secondPath)
    while(cap.isOpened()):
        # read frames
        ret, img = cap.read()

        # apply mask for background subtraction
        fgmask2 = fgbg2.apply(img)

        #save modifed images
        if i == 200:
            break
        imsave(secondPath + "/" + lis2[i], fgmask2)
        print(i)
        i += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
