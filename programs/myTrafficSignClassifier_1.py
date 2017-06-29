# LOAD THE DATA

#  Load pickled data
import pickle
import random
import cv2
import csv
import numpy as np
import os
from datetime import datetime



# TODO: Fill this in based on where you saved the training and testing data

training_file = "../traffic-signs-data/train.p"
validation_file = "../traffic-signs-data/valid.p"
testing_file = "../traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


#selected = np.random.choice(y_train.shape[0], 40000, replace=False)
#X_train = X_train[selected]
#y_train = y_train[selected]


# DATA SUMMARY
#
# ### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = sum(1 for line in open('signnames.csv'))


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# For the moment just like we did min-max normalization from 0.1-0.9

def normalize(image_data):
    return (image_data - 128.0) / 128.0
    # return image_data/(255.0)


def convert(image_data):
    for idx, x in enumerate(image_data):
        image_data[idx] = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)


# convert(X_train)
# convert(X_test)
# convert(X_valid)

X_train = normalize(X_train)
X_test = normalize(X_test)
X_valid = normalize(X_valid)

n_classes = 43

# Construim una llista de les feines a fer
treballs = [{'f_1': 20, 'f_2': 30, 'f_3': 100, 'f_4': 50, 'f_5': n_classes, 'k_p': 0.5, 'epochs':20}]

# Create a directory for the batch

directory = "./t_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = directory + "/"+"index.csv"

if not os.path.exists(directory):
    os.makedirs(directory)


print("Starting work storing data in "+ filename)

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ['F_1', 'F_2', 'F_3', 'F_4', 'F_5', 'K-Prob', 'Start', 'End', 'Time', 'Epochs', 'Best Epoch', 'Best accuracy',
         'Epochs'])

for iwork, work in enumerate(treballs):


    # Configuration

    f_1 = work['f_1'] # number of filters for first layer
    f_2 = work['f_2']  # number of filters for second layer
    f_3 = work['f_3']
    f_4 = work['f_4']
    f_5 = work['f_5']
    dropout = work['k_p']
    sizes = [f_1, f_2, f_3, f_4, f_5]
    EPOCHS = work['epochs']  # Just to test everything works :)
    BATCH_SIZE = 256

    # DATA EXPLORATION

    ### Data exploration visualization code goes here.
    ### Feel free to use as many code cells as needed.
    import matplotlib.pyplot as plt
    import numpy as np
    # Visualizations will be shown in the notebook.

    import csv
    #read the names
    names = []
    with open('signnames.csv', 'rt') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx > 0:
                names.append(row[1])



    ### Define your architecture here.
    ### Feel free to use as many code cells as needed.

    # First we will try LeNet. Just to see what happens


    import tensorflow as tf

    from tensorflow.contrib.layers import flatten


    def f(x):
        return tf.nn.tanh(x)


    def LeNet(x, dropout):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1



        weights = {
            'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, sizes[0]], mu, sigma)),
            'wc2': tf.Variable(tf.truncated_normal([5, 5, sizes[0], sizes[1]], mu, sigma)),
            'wd3': tf.Variable(tf.truncated_normal([sizes[1]*25, sizes[2]], mu, sigma)),
            'wd4': tf.Variable(tf.truncated_normal([sizes[2], sizes[3]], mu, sigma)),
            'wd5': tf.Variable(tf.truncated_normal([sizes[3], sizes[4]], mu, sigma))
        }

        biases = {
            'bc1': tf.Variable(tf.zeros(sizes[0])),
            'bc2': tf.Variable(tf.zeros(sizes[1])),
            'bc3': tf.Variable(tf.zeros(sizes[2])),
            'bc4': tf.Variable(tf.zeros(sizes[3])),
            'bc5': tf.Variable(tf.zeros(sizes[4]))
        }
        # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.

        conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1 = tf.nn.bias_add(conv1, biases['bc1'])

        # TODO: Activation.
        conv1 = f(conv1)
        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2 = tf.nn.bias_add(conv2, biases['bc2'])

        # TODO: Activation.
        conv2 = f(conv2)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Flatten. Input = 5x5x16. Output = 400.

        conv2 = flatten(conv2)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.

        fc3 = tf.add(tf.matmul(conv2, weights['wd3']), biases['bc3'])

        # TODO: Activation.
        fc3 = f(fc3)
        fc3 = tf.nn.dropout(fc3, dropout)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc4 = tf.add(tf.matmul(fc3, weights['wd4']), biases['bc4'])
        # TODO: Activation.
        fc4 = f(fc4)


        # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
        logits = tf.add(tf.matmul(fc4, weights['wd5']), biases['bc5'])

        return logits
    # Features and labels

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)
    keep_prob = tf.placeholder(tf.float32)

    ### Train your model here.
    ### Calculate and report the accuracy on the training and validation set.
    ### Once a final model architecture is selected,
    ### the accuracy on the test set should be calculated and reported as well.
    ### Feel free to use as many code cells as needed.

    rate = 0.001 #0.001

    logits = LeNet(x, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples


    from sklearn.utils import shuffle



    max_accuracy = 0.0
    max_test_accuracy = 0.0
    best_epoch = 0

    start_time = datetime.now()
    resultats = [f_1, f_2, f_3, f_4, f_5, dropout, start_time, 0, 0, EPOCHS, 0, 0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            validation_accuracy = evaluate(X_valid, y_valid)
            test_accuracy = evaluate(X_test, y_test)
            if validation_accuracy > max_accuracy:
                saver.save(sess, directory + '/' + str(iwork) + '_lenet')
                max_accuracy = validation_accuracy
                max_test_accuracy = test_accuracy
                best_epoch = i
                print("Model saved")

            resultats.append(validation_accuracy)

            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy),"Test Accuracy = {:.3f}".format(test_accuracy))
            print()

        #saver.save(sess, './lenet')
        print("Saved model validating accuracy = {:.3f}".format(max_accuracy))

        end_time = datetime.now()
        delta = end_time - start_time

        resultats[7] = end_time
        resultats[8] = delta
        resultats[10] = best_epoch
        resultats[11] = max_accuracy
        print("Starred at  : ", start_time , " Used ", delta)

        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(resultats)



