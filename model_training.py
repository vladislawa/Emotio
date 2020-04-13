import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn import preprocessing

df=pd.read_csv('fer2013.csv')

X_train, train_y, X_test, test_y = [], [], [], []

for index, row in df.iterrows():
	#row pixels is a string that are separated by spaces
	val = row["pixels"].split(" ") 
	try:
		if 'Training' in row["Usage"]:
			X_train.append(np.array(val, "float32"))
			train_y.append(row["emotion"])
		elif 'PublicTest' in row["Usage"]:
			X_test.append(np.array(val, "float32"))
			test_y.append(row["emotion"])
	except:
		print(f"Error occured at index:{index}, row: {row}")


#Converting train and test data to np arrays
X_train = np.array(X_train, "float32")
train_y = np.array(train_y, "float32")
X_test = np.array(X_test, "float32")
test_y = np.array(test_y, "float32")

#Normalizing data


#X_train -= np.std(X_train, axis=0)
#X_train /= np.std(X_train, axis=0)
#X_test -= np.std(X_test, axis=0)
#X_test /= np.std(X_test, axis=0)

X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48,48

#Reshaping data for keras
X_train = X_train.reshape(X_train.shape[0], width, height, 1)
X_test = X_test.reshape(X_test.shape[0], width, height, 1)

train_y = np_utils.to_categorical(train_y, num_classes = num_labels)
test_y = np_utils.to_categorical(test_y, num_classes = num_labels)




#Convolutional Neural Network
model = Sequential()
#First layer
model.add(Conv2D(num_features, kernel_size = (3,3), activation = "relu", input_shape=(X_train.shape[1:]))) #X_train.shape[1:] removing the number of rows
model.add(Conv2D(num_features, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2))) #we pick up the max value from the pool 
model.add(Dropout(0.5))

#Second layer
model.add(Conv2D(num_features, kernel_size = (3,3), activation = "relu"))
model.add(Conv2D(num_features, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2))) #we pick up the max value from the pool 
model.add(Dropout(0.5))

#Third layer
model.add(Conv2D(2*num_features, kernel_size = (3,3), activation = "relu"))
model.add(Conv2D(2*num_features, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2))) #we pick up the max value from the pool 

model.add(Flatten())

model.add(Dense(2*2*2*2*num_features, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2*2*2*2*num_features, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation = "softmax"))

model.compile(loss=categorical_crossentropy, optimizer = Adam(), metrics = ["accuracy"])

model.fit(X_train, train_y, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_test, test_y), shuffle=True)

fer_json = model.to_json()
with open ("emotion.json", "w") as json_file:
	json_file.write(fer_json)
model.save_weights("emotion.h5")


