import pickle
import numpy as np
import cv2
from datetime import datetime
import random
import matplotlib.pyplot as plt
import pydotplus as pydot
import numpy as np
import os
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
import keras.backend as K
from keras.utils import plot_model


import csv
#read the names
names = []
with open('signnames.csv', 'rt') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        if idx > 0:
            names.append(row[1])

# model_dir is where Keras saved the model. The name of file is model_dir+model.hf5
model_dir = "../New_Arc/keras_2017-06-25_15-25-51/"

# Name of the dataset from which images will be shown
dataset = "../traffic-signs-data/test.p"

# Name of the dataset which will be processed for the confusion maps
test_dataset = "../traffic-signs-data/test.p"


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    files = [] # Name of the files. Used to select some ones
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        f = True
        # gtReader.next() # skip header
        # # loop over all images in current annotations file
        for row in gtReader:
            if not f:

                raw_image = plt.imread(prefix + row[0])
                roi = raw_image[int(row[4]):int(row[6]), int(row[3]):int(row[5])]
                img = cv2.resize(roi, (28, 28))
                img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE);
                images.append(img) # the 1th column is the filename
                labels.append(int(row[7])) # the 8th column is the label
                files.append(prefix + row[0])
            else:
                f = False
        gtFile.close()
    return images, labels, files


def readTestTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    f = True
    prefix = rootpath + '/'  # subdirectory for class
    gtFile = open(prefix + 'GT-final_test.test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    f = True
    # gtReader.next() # skip header
    # # loop over all images in current annotations file
    for row in gtReader:
        if not f:

            raw_image = plt.imread(prefix + row[0])
            roi = raw_image[int(row[4]):int(row[6]), int(row[3]):int(row[5])]
            img = cv2.resize(roi, (28, 28))
            img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE);
            images.append(img) # the 1th column is the filename
            # labels.append(int(row[7])) # the 8th column is the label
        else:
            f = False
    gtFile.close()
    return images

# Selects n images from the GermanDataset and scales them as our signal


def getGermanDataset(n):

    all_images, all_labels, all_files = readTrafficSigns("../GTSRB/Final_Training/Images")

    return np.array(all_images), np.array(all_labels), np.array(all_files)

def getGermanTestDataset(n):

    all_images = readTestTrafficSigns("../GTSRB/Final_Test/Images")

    return np.array(all_images)

# Read images from a directory.

def get_images(path):

    images = []
    labels = []
    for file in os.listdir(path):
        if file[0] != ".":
            raw_image = plt.imread(path+"/"+file)
            img = cv2.resize(raw_image, (28, 28), interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE);
            images.append(img)  # the 1th column is the filename
            label = int(file.split("_")[0])
            labels.append(label)

    return np.array(images), np.array(labels)




def process(image_data):
    clahe = cv2.createCLAHE(2.0, (6, 6))

    for idx, x in enumerate(image_data):
        img = cv2.cvtColor(image_data[idx], cv2.COLOR_RGB2LAB)
        channels = cv2.split(img)

        img = cv2.equalizeHist(channels[0])
        img = clahe.apply(img)
        channels[0] = img
        img = cv2.merge(channels)
        img =  cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        aux = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.addWeighted(img, 1., aux, -0.9, 0)

        aux = cv2.GaussianBlur(img, (7,7), 0)
        img = cv2.addWeighted(img, 1., aux, 0.9, 0)

        image_data[idx] = img


def process_0(image_data):

    for idx, x in enumerate(image_data):
        img = cv2.cvtColor(image_data[idx], cv2.COLOR_RGB2LAB)
        channels = cv2.split(img)

        minv = np.min(channels[0])
        maxv = np.max(channels[0])
        mean = np.mean(channels[0])
        delta = max([maxv - mean, mean - minv])

        alfa = 128 / delta

        channels[0] = cv2.multiply(channels[0] ,alfa)
        img = cv2.merge(channels)

        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        image_data[idx] = img


def process_1(image_data):
    clahe = cv2.createCLAHE(2.0, (6, 6))

    for idx, x in enumerate(image_data):
        img = cv2.cvtColor(image_data[idx], cv2.COLOR_RGB2LAB)
        channels = cv2.split(img)

        img = cv2.equalizeHist(cv2.multiply(channels[0],1.0))
        img = clahe.apply(img)
        channels[0] = img
        img = cv2.merge(channels)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        image_data[idx] = img

def process_2(image_data):

    for idx, x in enumerate(image_data):
        img = image_data[idx]

        aux = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.addWeighted(img, 1., aux, -0.7, 0)


        image_data[idx] = img

def process_3(image_data):
    for idx, x in enumerate(image_data):
        img = image_data[idx]

        aux = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.addWeighted(img, 1., aux, 0.7, 0)

        image_data[idx] = img


def process_4(image_data):
    clahe = cv2.createCLAHE(2.0, (6, 6))
    for idx, x in enumerate(image_data):
        img = cv2.cvtColor(image_data[idx], cv2.COLOR_RGB2LAB)
        channels = cv2.split(img)

        img = cv2.equalizeHist(channels[0])
        img = clahe.apply(img)

        img = cv2.Canny(img, 150, 180)
        image_data[idx] = img.reshape(32,32,1)



### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry


# Modification by Paco Gorina.
# Seems it works. Also modified maximum number of rows depending of the size of the output
#


def outputFeatureMap(image_input, model, layer_n,  activation_min=-1, activation_max=-1, plt_num=1):
# def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):

    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function


    # activation = tf_activation.eval(session=sess, feed_dict={x: image_input})

    the_layer = model.get_layer(layer_n)
    t_output = the_layer.output
    t_input = the_layer.input

    sess = K.get_session()

    activation = t_output.eval(session=sess, feed_dict={model.input: image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 15))
    rows = featuremaps / 8 + 1
    for featuremap in range(featuremaps):
        pl = plt.subplot(rows, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
        pl.axis('off')

        #plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                       vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")

with open(dataset, mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# uncoment for reading from the German Dataset
# X_train, y_train, files = getGermanDataset(10)

# uncoment for reading from a directory
# X_train, y_train = get_images("images/german_web")


with open(test_dataset, mode='rb') as f:
    data = pickle.load(f)

X_valid, y_valid = data['features'], data['labels']

#X_valid = np.copy(X_train)
#y_valid = np.copy(y_train)


nimg=8

# Uncoment to user data from the train file loaded
#x_mean = np.mean(X_train)
#x_min = np.min(X_train)
#x_max = np.max(X_train)


# Copy resuts from super_train

x_mean = 41.6311489261
x_min = 0
x_max = 235

# OK Now we reinstantiate the model

model_pathname = model_dir + "model.hf5"
csv_pathname = model_dir + "allTest.csv"

model = load_model(model_pathname)

layers = model.layers

plot_model(model, to_file=model_dir+"/scheme.png")

for k in range(0, 1):

    selected = np.random.choice(X_train.shape[0], nimg, replace=False)

    # Uncoment to look at specific images
    #selected = np.array([0, 1, 2, 3, 4, 5, 6, 7])


    X_selected = X_train[selected]
    y_selected = y_train[selected]

    # Just in german Dataset uncoment to get the filenames selected
    #selected_files = files[selected]

    #for x in selected_files:
        #print(x)

    X_original = np.copy(X_selected)

    process(X_selected)

    X_normalized = np.array((X_selected - x_mean)/(x_max - x_min))

    # Now predict output for X_Train

    out = model.predict(X_normalized, verbose=1)
    predicted = np.zeros(nimg, dtype=np.int32)

    errors = 0

    for idx, pred in enumerate(out):
        i = np.argmax(pred, 0)

        predicted[idx] = i
        y = y_selected[idx]

        estat = "Correcte"
        if i != y:
            errors+= 1
            estat = "Erroni"

        print("Image:", i, "p:", names[predicted[idx]], "lab:", names[y_selected[idx]], estat)

    print("Errors:", errors, "of", nimg)


    XO_train = np.copy(X_original)

    fig, axes = plt.subplots(nrows=5, ncols=nimg)

    for i in range(0,nimg):
        axes[0, i].imshow(X_original[i])
        axes[0, i].axis('off')
        if predicted[i] == y_selected[i]:
            axes[0, i].set_title(predicted[i])
        else:
            axes[0, i].set_title(str(predicted[i]) + "/" + str(y_selected[i]))


    process_0(X_original)

    for i in range(0,nimg):
        axes[1, i].imshow(X_original[i])
        axes[1, i].axis('off')

    process_1(X_original)

    for i in range(0,nimg):
        axes[2, i].imshow(X_original[i])
        axes[2, i].axis('off')

    process_2(X_original)

    for i in range(0,nimg):
        axes[3, i].imshow(X_original[i])
        axes[3, i].axis('off')

    process_3(X_original)

    for i in range(0,nimg):
        axes[4, i].imshow(X_original[i])
        axes[4, i].axis('off')

    #for i in range(0,nimg):
    #    axes[4, i].imshow(XO_train[i])
    plt.subplots_adjust(wspace=0.02, hspace=0.01)
    plt.show()

# Show the feature map of last image???

image = np.array([X_normalized[0]])

# Output feature maps
outputFeatureMap(image, model, "conv2d_1",  activation_min=-1, activation_max=2, plt_num=1)
outputFeatureMap(image, model, "conv2d_2",  activation_min=-1, activation_max=2, plt_num=2)

plt.show()

# Comment exit to get the confusion maps and show the erroneus images

exit(0)

print("Processing validation set:" , test_dataset)
X_valid_org = np.copy(X_valid)
process(X_valid)
X_normalized = np.array((X_valid - x_mean)/(x_max - x_min))
out = model.predict(X_normalized, verbose=0)

bad_ones = []
index = 0

bad_freq = np.zeros((43, 43))

(histo, limits) = np.histogram(y_valid, bins=list(range(0,44)))

for (probs, y, img) in zip(out, y_valid, X_valid_org):
    pred = np.argmax(probs, 0)

    bad_freq[y][pred] = bad_freq[y][pred] + 1
    # print(pred, probs[pred], y, index)
    if pred != y:
        bad_ones.append([img, pred, y, probs[pred], index])

    index += 1

with open(csv_pathname, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    l0 = list(range(0,43))
    l1 = ['Dataset', 'test_dataset', '_', '_', 'Net']
    l1.extend(l0)
    csvwriter.writerow(l1)

    l1 = ['N', 'Label', 'Total', 'Errors', '%']
    l1.extend(names)
    csvwriter.writerow(l1)


    for index, freq in enumerate(bad_freq):
        name = names[index]
        tot = np.sum(freq) - freq[index] # errors = total - correct ones
        l1 = [str(index), name, str(histo[index]), str(tot), str(tot/histo[index])]
        l1.extend(freq)
        csvwriter.writerow(l1)

        # print(index, " - ", name ,":",str(tot),freq)

nrows = int(len(bad_ones) / 10 + 1)

fig, axes = plt.subplots(nrows=min(nrows,6), ncols=10)

row = 0

maxrow = 6

for i, bad in enumerate(bad_ones):

    img = bad[0]
    bad_v = bad[1]
    ok_v = bad[2]
    pr = bad[3]
    axes[int(i/10)%6, i%10].imshow(img)
    axes[int(i/10)%6, i%10].axis('off')
    axes[int(i/10)%6, i%10].set_title(str(bad_v) + " / "  + str(ok_v))

    if int(i % 60) == 0 and i != 0:
        plt.show()
        fig, axes = plt.subplots(nrows=min(nrows,6), ncols=10)



