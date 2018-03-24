import numpy as np
import cv2 as cv
import PIL as image
import os

#This is the path to the KTH folder
path = r"C:\Users\liuyu\Desktop\Bigdata\KTH\frame"

directory = r"C:\Users\liuyu\Desktop\Bigdata\KTH\boxing"

for file in os.listdir(directory):
    #Using count to rename the frame extract from the video
    count = 0
    filename = file[:-11]
    print("Begin subtract background for {}!".format(filename))
    try:
        os.mkdir(os.path.join(path,filename))
    except:
        print("Create folder{} fail!".format(filename))

    video = cv.VideoCapture(os.path.join(directory,file))

    #Another way to do the background subtraction
    #fgbg = cv.createBackgroundSubtractorKNN()

    fgbg = cv.createBackgroundSubtractorMOG2()

    while True:
        success,frame = video.read()
        if success == True:
            frame_after_extract = fgbg.apply(frame)
            if count!= 0:
                cv.imwrite(os.path.join(os.path.join(path,filename),filename+"_{}.jpg".format(count)),frame_after_extract)
            count += 1

        elif success == False:
            break

    print("Finish subtract background for {}!".format(filename))









