import cv2 as cv
import numpy as np
import os
from time import sleep

cap = cv.VideoCapture(0)
   
num = 0
while True:

    _, frame = cap.read()
    frame = cv.flip(frame,1)
    frame = cv.resize(frame, (1000,600))

    ####### Rectangle and texts

    cv.rectangle(frame, (500, 200),(750,500), color=(0,255,0), thickness=4)
    cv.putText(frame, "PUT YOUR HAND HERE",(500,530),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv.putText(frame, "a: ROCK",(500,550),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv.putText(frame, "s: PAPER",(600,550),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv.putText(frame, "d: SCISSOR",(700,550),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv.putText(frame, "f: NONE",(800,550),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    roi = frame[203:498, 503:747]
    
    ###### Start taking images
    
    ### ROCK
    if cv.waitKey(2) == 65 or cv.waitKey(2) == 97:
        os.chdir('C:\\Users\\user\\Documents\\Python Scripts\\Machine Learning\\Rock Paper Scissor\\images\\ROCK\\')
        cv.imwrite("ROCK{}.jpg".format(num), roi)
        num += 1
    
    ### PAPER
    if cv.waitKey(2) == 115:
        os.chdir('C:\\Users\\user\\Documents\\Python Scripts\\Machine Learning\\Rock Paper Scissor\\images\\PAPER\\')
        cv.imwrite("PAPER{}.jpg".format(num), roi)
        num += 1

    ### SCISSOR
    if cv.waitKey(2) == 100:
            os.chdir('C:\\Users\\user\\Documents\\Python Scripts\\Machine Learning\\Rock Paper Scissor\\images\\SCISSOR\\')
            cv.imwrite("SCISSOR{}.jpg".format(num), roi)
            num += 1

    ### NONE
    if cv.waitKey(2) == 102:
        os.chdir('C:\\Users\\user\\Documents\\Python Scripts\\Machine Learning\\Rock Paper Scissor\\images\\NONE\\')
        cv.imwrite("NONE{}.jpg".format(num), roi)
        num += 1


    if num != 0:
        cv.putText(frame, str(num),(300,330),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    if cv.waitKey(1) == 114:
        num = 0


    ############## Show and waitkey
    if cv.waitKey(2) == 27:
        break
    cv.imshow("vid", frame)

cv.destroyAllWindows()
