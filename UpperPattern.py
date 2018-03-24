import numpy as np
import cv2 as cv
from PIL import Image
import os
import statistics

#This is the path to the result folder
path = r"C:\Users\liuyu\Desktop\Bigdata\KTH\frame"

#This is the path to the video to original video folder
directory = r"C:\Users\liuyu\Desktop\Bigdata\KTH\boxing"

def XOR(a,b):
    return a!=b

def get_xba(n,p):
    xba = []
    width,height= np.size(p)
    fp = p.load()
    fn = n.load()
    #Get center pixel values
    fp_c = fp[(width-1/2),(height-1/2)]
    fn_c = fn[(width-1/2),(height-1/2)]

    fpminusc = []
    fnminusc = []
    for i in range(width):
        for j in range(height):
            fpminusc.append(fp[i,j]-fp_c)
            fnminusc.append(fn[i,j]-fn_c)

    fpminusc.sort()
    fnminusc.sort()
    xpba = statistics.median(fpminusc)
    xnba = statistics.median(fnminusc)

    xba.append(fp_c)
    xba.append(fn_c)
    xba.append((xpba+xnba)/2)

    return xba

def Upper_ALMD(p,n,xba):

    def Uq(a):
        return a>=xba[0]

    fp = p.load()
    fn = n.load()

    width,height = np.size(p)

    sum = 0
    l = 0

    for i in range(width):
        for j in range(height):
            sum += XOR(Uq(fp[i,j]-xba[1]),Uq(fn[i,j]-xba[1]))*2^(l)
            l+=1

    return sum

def Lower_ALMD(p,n,xba):


    return

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

    V = []
    Q = []
    while True:
        successp,framep = video.read()
        successn,framen = video.read()
        if successp == True and successn == True:
            frame_p = fgbg.apply(framep)
            frame_n = fgbg.apply(framen)

            if count!= 0:
                cv.imwrite(os.path.join(os.path.join(path,filename),filename+"_{}.jpg".format(count)),frame_after_extract)
            count += 1
            xba = get_xba(frame_p,frame_n)
            V.append(Upper_ALMD(frame_p,frame_n,xba))
            Q.append(Lower_ALMD(frame_p,frame_n,xba))

        else:
            break

    print("Finish subtract background for {}!".format(filename))

if __name__ == "__main__":
    frame1 = Image.open(r"C:\Users\liuyu\Desktop\Bigdata\KTH\frame\person03_walking_d1\person03_walking_d1_1.jpg")
    frame2 = Image.open(r"C:\Users\liuyu\Desktop\Bigdata\KTH\frame\person03_walking_d1\person03_walking_d1_2.jpg")

    xba = get_xba(frame1,frame2)
    V= []
    V.append(Upper_ALMD(frame1,frame2,xba))
    print(V)

