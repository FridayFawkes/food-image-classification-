"""
Test model training
"""
#Please start from here!!
###############################################################################
#                           Build a CNN model                                 #
###############################################################################
#CNN model
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

# Test fix for cuDNN STATUS INTERNAL ERROR
physical_devices=tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))

model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #initial lr = 0.01
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print(model.summary())

####################################
#         Callback Schedule       #
###################################
from tensorflow import keras
class decaylr_loss(tf.keras.callbacks.Callback):
    def __init__(self):
        super(decaylr_loss, self).__init__()
    def on_epoch_end(self,epoch,logs={}):
        #loss=logs.items()[1][1] #get loss
        loss=logs.get('loss')
        print( "loss: ",loss)
        old_lr = 0.001 #needs some adjustments
        new_lr= old_lr*np.exp(loss) #lr*exp(loss)
        print( "New learning rate: ", new_lr)
        K.set_value(self.model.optimizer.lr, new_lr)
lrate = decaylr_loss()
#early stopping
patience = 20
earlystopper = EarlyStopping(monitor='val_acc', patience=patience, 
                             verbose=1, mode='max')       
#check point
wdir = './wdir' #work directory
filepath = os.path.join(wdir,'modelWeights','cnnModelDEp80weights.best.hdf5') #save model weights
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

###############################################################################
#                     Data Expansion or Augmentation                          #
###############################################################################
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                   #featurewise_center=True,
                                   #featurewise_std_normalization=True,
                                   rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

trainDir = './resizedData/Train'
train_generator = train_datagen.flow_from_directory(trainDir,  
                                                    target_size=(64,64), #ORIGINAL (128, 128)
                                                    batch_size=32,
                                                    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)  
testDir = './resizedData/Test'
test_generator = test_datagen.flow_from_directory(testDir,
                                                  target_size=(64,64), #ORIGINAL (128, 128)
                                                  batch_size=32,
                                                  shuffle=False,
                                                  class_mode='categorical')

input("Datasets checkpoint. Press any key to continue...\n")

###############################################################################
##                      Fit, Evaluate and Save Model                          #                                  
###############################################################################
#epochs = 100
#epochs = 200
epochs = 400
#epochs = 600
samples_per_epoch = 4654 
val_samples = 1168

#Fit the model
hist = History()
model.fit_generator(train_generator,
                    steps_per_epoch= (samples_per_epoch/32),
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_generator,
                    callbacks = [earlystopper, lrate, checkpoint, hist])

#evaluate the model
scores = model.evaluate_generator(test_generator) 
print("Accuracy = ", scores[1])

#save model
savePath = wdir
model.save_weights(os.path.join(savePath,'cnnModelDEp80.h5')) # save weights after training or during training
model.save(os.path.join(savePath,'cnnModelDEp80.h5')) #save complied model

#plot acc and loss vs epochs
import matplotlib.pyplot as plt
print(hist.history.keys())
#accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig(os.path.join(savePath,'cmdeP80AccVsEpoch.jpeg'), dpi=1000, bbox_inches='tight')
# plt.show()
#loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig(os.path.join(savePath,'cmdeP80LossVsEpoch.jpeg'), dpi=1000, bbox_inches='tight')
# plt.show()

# ###############################################################################                          
# #Note: train 4364 images (80%) and test 1458 images (20%)                     # 
# # 100 epochs:                                                                 #
# # 200 epoches: acc = 0.8724; val_acc = 0.89212                                #  
# # 400 epoches:                                                                #
# # 600 epochs:                                                                 #
# ###############################################################################

# # load the model
# # not necessary the best at the end of training 
# from tensorflow.keras.models import load_model
# myModel = load_model(os.path.join(savePath,'cnnModelDEp80.h5'))                                  
# scores = myModel.evaluate_generator(test_generator)
# print("Accuracy = ", scores[1])

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

