from keras.models import load_model 
import cv2 as cv 
import numpy as np

REV_CLASS_MAP = {
    1: "ROCK",
    2: "PAPER",
    3: "SCISSOR",
    0: "none"
}

model = load_model('ROCK-PAPER-SCISSOR-TRAINED.h5')
    
img = cv.imread('PAPER.jpg')
img = cv.resize(img,(227,227))
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

pred = model.predict(np.array([img]))

print("pridected: ", REV_CLASS_MAP[np.argmax(pred[0])])

cv.destroyAllWindows()