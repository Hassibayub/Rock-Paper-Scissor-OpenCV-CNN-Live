import cv2 as cv
import numpy as np
import os
import keras
from random import choice
from time import sleep

cap = cv.VideoCapture(0)
model = keras.models.load_model('ROCK-PAPER-SCISSOR-TRAINED.h5')

REV_MODEL_CLASS = {
    0 : 'NONE',
    1 : 'ROCK',
    2 : 'PAPER',
    3 : 'SCISSOR' 
}

def ROBO_MOVE_EMOJI(var):
    img = cv.imread('EMOJI//{}.png'.format(var))
    return img

def decideWin(human, robo):
    # print("human move: ", human)
    # print("robo move: ", robo)

    if (human == 'ROCK' and robo =='SCISSOR'):
        return 'YOU WIN'
    elif (human == 'PAPER' and robo =='ROCK'):
        return 'YOU WIN'
    elif (human == 'SCISSOR' and robo =='PAPER'):
        return 'YOU WIN'

    elif (robo == 'ROCK' and human =='SCISSOR'):
        return 'YOU LOSS'
    elif (robo == 'PAPER' and human =='ROCK'):
        return 'YOU LOSS'
    elif (robo == 'SCISSOR' and human =='PAPER'):
        return 'YOU LOSS'
    elif (human == robo):
        return 'TIE'
    else:
        return 'WAITING..'

    # pass


LAST_MOVE = ''
robo_move_name = ''

while True:

    _, frame = cap.read()
    frame = cv.flip(frame,1)
    frame = cv.resize(frame, (1000,600))


    ########### YOUR RECTANGLE
    cv.rectangle(frame, (700, 200),(950,500), color=(0,255,0), thickness=4)
    cv.putText(frame, "YOUR MOVE",(700,530),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    ############ ROBOT RECTANGLE
    cv.rectangle(frame, (100,200), (350,500), color=(0,255,0),thickness=4)
    cv.putText(frame, "ROBOT",(100,530),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    roi = frame[203:497, 700:950]

    ########### YOU DECIDE
    roi = cv.resize(roi,(227,227))
    roi = cv.cvtColor(roi,cv.COLOR_BGR2RGB)
    # roi = np.array(roi)
    pred = model.predict(np.array([roi]))
    move_name = REV_MODEL_CLASS[np.argmax(pred[0])]
    cv.putText(frame, str(move_name) ,(600,150),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

    ########### ROBOT TURN
    # print("move_name: ",move_name)


    if move_name != LAST_MOVE:
        if (move_name != "NONE"):
            robo_move = choice([1,2,3])
            # print("robo_move: ", robo_move)
            robo_move_name = REV_MODEL_CLASS[robo_move]
            robo_move_img = ROBO_MOVE_EMOJI(robo_move_name)
            robo_move_img = cv.resize(robo_move_img,(250,294))  # latest change, not yet tested
            cv.imshow("robo",robo_move_img)
            
            LAST_MOVE = move_name

    
    winner = decideWin(move_name,robo_move_name)

    if (winner != 'NONE'):
        print("the winner is ", winner)


    if cv.waitKey(2) == 27:
        break
    cv.imshow("vid", frame)


    

cv.destroyAllWindows()