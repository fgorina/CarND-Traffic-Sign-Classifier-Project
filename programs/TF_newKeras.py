# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime
import os

tf.python.control_flow_ops = tf

dataset = "../traffic-signs-data/super_train.p"

with open(dataset, mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

with open("../traffic-signs-data/valid.p", mode='rb') as f:
    data = pickle.load(f)

X_valid, y_valid = data['features'], data['labels']


def process(image_data):
    clahe = cv2.createCLAHE(2.0, (6, 6))

    for idx, x in enumerate(image_data):

        img = cv2.cvtColor(image_data[idx], cv2.COLOR_RGB2LAB)
        channels = cv2.split(img)

        img = cv2.equalizeHist(channels[0])
        img = clahe.apply(img)
        channels[0] = img
        img = cv2.merge(channels)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        aux = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.addWeighted(img, 1., aux, -0.9, 0)

        aux = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.addWeighted(img, 1., aux, 0.9, 0)

        image_data[idx] = img

# Initial Setup for Keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from sklearn.preprocessing import LabelBinarizer

# Define filename and create directory

directory = "./keras_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
weights_filename = directory + "/"+"weights.hf5"
# model_filename = directory+"/"+"model_{epoch:02d}.hf5"
model_filename = directory+"/"+"model.hf5"

if not os.path.exists(directory):
    os.makedirs(directory)

# Create a callback to save best fit

k_callback = ModelCheckpoint(model_filename, monitor='val_acc', verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)


# TODO: Build Convolutional Neural Network in Keras Here

f_1 = 60
f_2 = 100
f_21 = 250
f_22 = 250
f_3 = 200
f_4 = 100
f_5 = 43
dropout = 0.5
use_valid_data = True
lr = 0.0003
reg = 0.005
epochs = 40

model = Sequential()

model.add(Conv2D(f_1, (5, 5), input_shape=(32, 32, 3),
                 activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(f_2, (3, 3),   activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(f_21, (3, 3),   activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))

# Added new layes. Results????
model.add(Conv2D(f_22, (3, 3),   activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(f_3, kernel_initializer='truncated_normal',
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(f_4, kernel_initializer='truncated_normal',
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(f_5, kernel_regularizer=regularizers.l2(0.005),
                kernel_initializer='truncated_normal',
                bias_initializer='zeros'))
model.add(Activation('softmax'))

# Preprocess data

process(X_train)
process(X_valid)


# Get normalization values

x_mean = np.mean(X_train)
x_min = np.min(X_train)
x_max = np.max(X_train)
X_normalized = np.array((X_train - x_mean)/(x_max - x_min))
X_normalized, y_train = shuffle(X_normalized, y_train)
X_valid_normalized = np.array((X_valid - x_mean)/(x_max - x_min))

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
y_valid_one_hot = label_binarizer.fit_transform(y_valid)

print("Dataset", dataset)
print("Using Image Processing two Gaussians, +/- 0.9")
print("Saving model data to " + directory)
print("Using Validation Data", use_valid_data)
print("Regularization at Dense(f_5 0.01)")
print("Sizes", f_1, f_2, f_21, f_22, f_3, f_4, f_5, "Dropout", dropout, "Regularization", reg, "same")
print("Learning Rate ", lr)
print("Normalization Mean", x_mean, "min: ", x_min, "max: ", x_max)


adam = Adam(lr=lr)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

if use_valid_data:
    history = model.fit(X_normalized, y_one_hot, epochs=epochs, verbose=2, validation_data=(X_valid_normalized, y_valid_one_hot), callbacks=[k_callback])
else:
    history = model.fit(X_normalized, y_one_hot, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[k_callback])

with open("../traffic-signs-data/test.p", 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
process(X_test)
X_normalized_test = np.array((X_test - x_mean)/(x_max-x_min))
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

model = load_model(model_filename)
# model.compile('adam', 'categorical_crossentropy', ['accuracy'])

# TODO: Evaluate the test data in Keras Here
metrics = model.evaluate(X_normalized_test, y_one_hot_test)
# TODO: UNCOMMENT CODE
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
