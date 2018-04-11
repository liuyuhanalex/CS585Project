import numpy as np
import cv2 as cv
import PIL as image
import os
import datetime

#This is the path to the KTH folder
path = r"C:\Users\liuyu\Desktop\Bigdata\KTH\frame"

directory = r"C:\Users\liuyu\Desktop\Bigdata\KTH\boxing"

testfile = open(r'C:\Users\liuyu\Desktop\pop.csv','w+')

#size
#Resize the frame and only use the first 100 frames
width = 40
height = 30
#generate function for test
def generate(col,row):
    A =[]
    for i in range(row):
        A.append([np.random.randint(0,256) for j in range(col)])
    return A

def ALMD(pre_frame,next_frame):
    Upper_ALMD = []
    Lower_ALMD = []
    pre = np.reshape(pre_frame,(width,height))
    nex = np.reshape(next_frame,(width,height))
    for i in range(1,width-1):
        for j in range(1,height-1):
            g_pc = pre[i][j]
            xp_array = [pre[i-1][j-1],pre[i][j-1],pre[i+1][j-1],pre[i][j-1],pre[i][j+1],
            pre[i-1][j+1],pre[i][j+1],pre[i+1][j+1]]
            x_p = np.median(np.sort([int(x)-int(g_pc) for x in xp_array]))
            g_nc = nex[i][j]
            xn_array = [nex[i-1][j-1],nex[i][j-1],nex[i+1][j-1],nex[i][j-1],nex[i][j+1],
            nex[i-1][j+1],nex[i][j+1],nex[i+1][j+1]]
            nl_nc = np.zeros(8)
            pl_nc = np.zeros(8)
            for p in range(8):
                nl_nc[p] = int(xn_array[p]) - int(g_nc)
                pl_nc[p] = int(xp_array[p]) - int(g_nc)
            x_n = np.median(np.sort(nl_nc))
            x_ba = (x_p+x_n)/2
            upper_sum = 0
            lower_sum = 0
            for m in range(8):
                upper_sum += ((pl_nc[m]>x_ba)!=(nl_nc[m]>x_ba))*(2**m)
                lower_sum += (((pl_nc[m]<(-x_ba)))!=(nl_nc[m]<(-x_ba)))*(2**m)
            Upper_ALMD.append(upper_sum)
            Lower_ALMD.append(lower_sum)
    Result=[Upper_ALMD,Lower_ALMD]
    return Result
begintime = datetime.datetime.now()
for file in os.listdir(directory):
    #Using count to rename the frame extract from the video
    count = 0
    filename = file[:-11]
    V=[]
    Q=[]
    print("Begin subtract background for {}!".format(filename))
    try:
        os.mkdir(os.path.join(path,filename))
    except:
        print("Create folder{} fail!".format(filename))

    video = cv.VideoCapture(os.path.join(directory,file))

    #Another way to do the background subtraction
    #fgbg = cv.createBackgroundSubtractorKNN()

    fgbg = cv.createBackgroundSubtractorMOG2()

    while count<100:
        success,frame = video.read()
        if success == True:
            next_frame = fgbg.apply(frame)
            next_frame = cv.resize(next_frame,(40,30))
            cv.imwrite(os.path.join(os.path.join(path,filename),filename+"_{}.jpg".format(count)),next_frame)
            if count!= 0:
                Result=ALMD(pre_frame,next_frame)
                V.append(Result[0])
                Q.append(Result[1])
            else:
                success1,frame1 = video.read()
                pre_frame = fgbg.apply(frame1)
                pre_frame = cv.resize(pre_frame,(40,30))
                cv.imwrite(os.path.join(os.path.join(path,filename),filename+"_{}.jpg".format(count)),pre_frame)
            count += 1
            if count!=1:
                pre_frame = next_frame

        else:
            break

    print("Finish subtract background for {}!".format(filename))
    testfile.writelines(filename.split('_')[1]+'|')
    testfile.writelines(str(V)+'|')
    testfile.writelines(str(Q)+'\n')
endtime = datetime.datetime.now()
var = endtime - begintime
print("begin time is "+ str(begintime))
print("end time is "+ str(endtime))
print("var is "+ str(var))

#Test part
# A1 = generate(160,120)
# A2 = generate(160,120)
# Result = ALMD(A1,A2)
# print(np.size(Result[0]))
