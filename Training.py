import numpy as np
import keras 
from keras_squeezenet import SqueezeNet
import os 
import cv2 as cv 
import tensorflow as tf

FILE_DIR = r'C:\Users\user\Documents\Python Scripts\Machine Learning\Rock Paper Scissor\images'

rock = 0
paper = 0
scissor = 0
none = 0

get_label = {
    "NONE" : 0,
    "ROCK" : 1,
    "PAPER": 2,
    "SCISSOR": 3
}

NUM_CLASS = len(get_label)

def get_value_along_label(val):
    return get_label[val]


counter = 0
dataset=[] # dataset[ [image array],'paper']
for roots ,folders , files in os.walk(FILE_DIR):
    for file_ in files:
        # print(os.path.join(roots,file_))
        path = os.path.join(roots,file_)
        img = cv.imread(path)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img = cv.resize(img,  (227,227))
        dataset.append([img,os.path.basename(roots)])
        counter += 1

print("################################################### COUNTER")
print(counter)

image , labels = zip(*dataset)
label = list(map(get_value_along_label,labels))

labels = keras.utils.np_utils.to_categorical(label)

model = keras.models.Sequential([
    SqueezeNet(input_shape = (227,227,3), include_top = False), 
    keras.layers.Dropout(0.5), 
    keras.layers.Convolution2D(NUM_CLASS, (1,1), padding='valid'),
    keras.layers.Activation('relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Activation('softmax')
])

model.compile(
    optimizer = keras.optimizers.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy']
)

model.fit(np.array(image), np.array(labels),epochs=10)

model.save('ROCK-PAPER-SCISSOR-TRAINED.h5')