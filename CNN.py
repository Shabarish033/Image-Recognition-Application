  # -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:44:56 2019

@author: shabarish
"""
#My CNN
#%%Importing Libraries
import os
import numpy as np
from keras.models import Sequential # Initialize the Neural Network
from keras.layers import Convolution2D #Create Convolution layer
from keras.layers import MaxPooling2D #Create max pooling layer
from keras.layers import Flatten #convert matrix to vectors
from keras.layers import Dense #Connect the layers
from keras.preprocessing.image import ImageDataGenerator #For image Augmentation
from keras.preprocessing import image
from keras.models import model_from_json #To save the model 
#%%Data is already preprocessed, image files are stored in respective paths
#Extracting Data
TrainingSetPath =  os.environ['TrainingSetPath']
TrainingSetPath = str(TrainingSetPath)
TestSetPath =  os.environ['TestSetPath']
TestSetPath = str(TestSetPath)
path1 = os.environ['path1']
path1 = str(path1)
#%%Initialize the CNN
classifier = Sequential()
#First Convolution and MaxPooling Layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu')) #Added Convolution Layer
classifier.add(MaxPooling2D(pool_size=(2, 2))) #Added Max Pooling Layer
#%%Second Convolution and MaxPooling Layer
classifier.add(Convolution2D(32, 3, 3, activation='relu')) #Added Convolution Layer
classifier.add(MaxPooling2D(pool_size=(2, 2))) #Added Max Pooling Layer
classifier.add(Flatten())#converting image matrices to vectors
classifier.add(Dense(output_dim=128, activation='relu')) #Full Connection
classifier.add(Dense(output_dim=1, activation='sigmoid')) #Output Layer
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%Fitting CNN
ImageModificationsTrainingSet = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

ImageModificationsTestSet = ImageDataGenerator(rescale=1./255)

TrainingSet = ImageModificationsTrainingSet.flow_from_directory(
        os.path.join(path1, 'TrainingSet'),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
 
TestSet = ImageModificationsTestSet.flow_from_directory(
        os.path.join(path1, 'TestSet'),
        target_size=(64, 64),
        batch_size=12,
        class_mode='binary')

classifier.fit_generator(
        TrainingSet,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=TestSet,
        validation_steps=2000)


