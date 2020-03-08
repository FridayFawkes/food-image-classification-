# Script to load and make predictions with the model

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# test data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Path to the saved model
savePath = './savedModels'

# 64x64 Model
test_datagen = ImageDataGenerator(rescale=1./255)  
testDir = './resizedData/Test'
test_generator = test_datagen.flow_from_directory(testDir,
                                                  target_size=(64,64),
                                                  batch_size=32,
                                                  shuffle=False,
                                                  class_mode='categorical')


# 128x128 Model
# testDir = './Data/Test'
# test_generator = test_datagen.flow_from_directory(testDir,
#                                                   target_size=(128,128),
#                                                   batch_size=32,
#                                                   shuffle=False,
#                                                   class_mode='categorical')


# load the model
# not necessary the best at the end of training 

# myModel = load_model(os.path.join(savePath,'cnnModelDEp80.h5'))                                  

# 64x64 Model
myModel = load_model(os.path.join(savePath,'VGGModel-64x64-400epochs-73acc.h5'))

# 128x128 Model
# myModel = load_model(os.path.join(savePath,'VGGModel-128x128-400epochs-68acc.h5'))

scores = myModel.evaluate_generator(test_generator)
print("Accuracy = ", scores[1])

# ########################
# # Check-pointed model  #
# #######################
# model = Sequential()
# model.add(Convolution2D(filters=32, kernel_size=(7, 7), input_shape=(128, 128, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(64, 5, 5))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(128, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten()) 
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))
# lr = 0.00277615583366 #adjust lr based on training process
# #lr = 0.00181503843077
# #lr = 0.00163685841542 #case 2
# #lr =  0.00122869861281 # case3
# sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) 
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
# model.load_weights(filepath) #load saved weights
# scores = model.evaluate_generator(test_generator)
# print("Accuracy = ", scores[1])


# #Confusion matrix on the test images
# #imgDir = testDir
# imgDir = trainDir
# test_generator = test_datagen.flow_from_directory(imgDir,
#                                                   target_size=(128,128),
#                                                   batch_size=32,
#                                                   shuffle=False, 
#                                                   class_mode='categorical')
# #val_samples = 1168
# val_samples = 4654
# predict = model.predict_generator(test_generator)

# yTrue = test_generator.classes
# yTrueIdx = test_generator.class_indices

# from sklearn.metrics import classification_report, confusion_matrix
# yHat = np.ones(predict.shape[0],dtype = int)
# for i in range(predict.shape[0]):
#     temp = predict[i,:]
#     yHat[i] = np.argmax(temp)  
    
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(yTrue,yHat)
# print( "Accuracy on test images:", acc) #same as scores[1]

# def numToLabels(y,cat):
#     numLabel = []
#     import numpy as np
#     yNew = np.unique(y) #sorted 
#     for i in range(len(y)):
#         idx = np.where(yNew == y[i])[0][0]
#         numLabel.append(cat[idx])
#     return numLabel                   
# #labels = sorted(yTrueIdx.keys())
# labels = ['Ap','Ba','Br','Bu','Eg','Fr','Hd','Pz','Rc','St']
# yActLabels = numToLabels(yTrue,labels)
# yHatLabels = numToLabels(yHat,labels)
# CM = confusion_matrix(yActLabels,yHatLabels,labels) #np.array
# #print CM    
# print(classification_report(yTrue,yHat,target_names=labels))

# #Alternatively: pd.crosstab
# import pandas as pd
# #preds = pd.DataFrame(predict)
# y1 = pd.Categorical(yActLabels,categories=labels)
# y2 = pd.Categorical(yHatLabels,categories=labels)
# pd.crosstab(y1,y2,rownames=['True'], colnames=['Predicted'])

