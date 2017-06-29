recognizing#** Traffic Sign Recognition **

---

** Build a Traffic Sign Recognition Project **


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[Dataset_data]: ./images/dataset_frequencies.png
[Example_images]: ./images/example_images.png
[Image_processing_1]: ./images/image_processing.png
[Image_processing_2]: ./images/image_processing_2.png
[Confusion]: ./images/confusion.png "Confusion"
[Loss]: ./images/loss.png "Loss"
[Accuracy]: ./images/accuracy.png "Accuracy"
[image1]: ./images/german_web/01_Speed_limit_30.jpg
[image2]: ./images/german_web/4_Speed_70.jpg
[image3]: ./images/german_web/11_rightofway.jpg
[image4]: ./images/german_web/17_no_entry.jpg
[image5]: ./images/german_web/18_General_Caution.jpg
[image6]: ./images/german_web/25_Road_work.jpg
[image7]: ./images/german_web/27_pedestrians.jpg
[image8]: ./images/german_web/33_Turn_Right_Ahead.jpg
[prediction]: ./images/prediction.png
[original]: ./images/original.png
[layer_1]: ./images/layer_1.png
[layer_2]: ./images/layer_2.png
## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb) and the [folder with aditional code](./programs)

## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set.

I used numpy to analyze the data sets and I included my augmented data set. This is just a simple analysis counting the number of samples in each data set and the number of classes from the signnames.csv file.

* The size of the training set is 34799 samples
* The size of the augmented training set is 104354 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the frequency of the different classes for each set. They are similar but not identical.

![Dataset frequency analysis][Dataset_data]

The size of the sets is very different, as may be expected although the small size of the validation  set may be doing some strange effects that will be commented later.

Here you have an example of the images present in the dataset:

![Example images][Example_images]

As a comment, not only the number of images for class are important, but also the variety of them. I suppose that limited variety and the small validation set size produces results over the validation set much better than over the test set.

That was made clear to me in the first augmentations of the training set. An error eliminated rotations and many errors in classification where caused from this. Once corrected misidentifications due to turned signals where much less frequent.


## Design and Test a Model Architecture

### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Well, never sold about only gray images.

I did some first tests, with grayscale, converting to a LAB color encoding and had not better results than with color.

True is that then I had no access to a GPU and also models were simpler but was not really convinced.

Finally I set on a pipeline that :

* Converts image to LAB
* Splits the channels and applies
    * **equalizeHist** to first channel
    * **CLAHE** with a 6x6 tile grid size to first channel
* Merge channels and convert back to RGB
* Apply two **Gaussian blur**
    * First with a kernel of 5x5 and doing a weighted subtraction
    * Second with a kernel of 7x7 and doing a weighted addition


Here are some examples of the pipeline

![Image Processing][image_processing_1]

Finally normalization was applied with the idea of centering all values around 0 ad having less variance.

To do it I computed the mean, maximum and minimum values of the training set and used them to normalize data according:

    x_mean = np.mean(X_train)
    x_min = np.min(X_train)
    x_max = np.max(X_train)

    def normalize(image_data):
        return np.array((image_data - x_mean)/(x_max - x_min))


Finally after much tests I decided to augment the data set by incorporating transformations of the original images.

The augmentation was done offline and created the new ** super_train.p ** dataset.

I tried two approaches, first equalize the frequencies of all classes, but it seemed a bit ad hoc as some images will be mainly modifications of a small number of real images.

Final approach used a random selection of images so it mimics the original frequency of the train set but multiplies its size by 3.

The algorithm used was implemented in [new_generate.py](./programs/new_generate.py)
and is the following:

* Apply a random -0.5 to 1.5 factor to the L channel of the image, changing its lightness
* Apply a random rotation between -20 to +20 degrees
* Apply a random resize giving images between 18x10 and 32x32 pixels, and filling the border to get 32x32 pixels again
* Apply random translations between +5, -5 pixels to each, x and y directions filling the borders once again

Then the file is written to disk as super_train.py.

|               |Original train.p| Augmented super_train.p|
| :--------------|:-------------- | :--------------------- |
| Size (Mb)      | 107.1          | 320.7                  |
| Examples       | 34799          | 104354                 |




### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Half of the development was made with tensorflow directly. Mainly analyzing different size of LeNet layers and testing functions like **tanh** instead of **relu**. For some time it seemed has better convergence or at least faster.

After I changed to a Keras based approach. It was much easier to make modifications and as I gained access to a GPU things evolved quickier or at least I could test more architectures.

I tried different approaches, as injecting first layer data into the classifier but finally decided the following architecture:


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x60 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x60 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x100 			    	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x250 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x14x250 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x250 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x250  				|
| Flatten				| outputs 1000									|
| Fully connected       | Outputs 200                                   |
| RELU                  |                                               |
| dropout               | Keep Probability 0.5                          |
| Fully connected       | Outputs 100                                   |
| RELU                  |                                               |
| dropout               | Keep Probability 0.5                          |
| Fully connected       | Outputs 43                                    |
| softmax               | Outputs 43                                    |
| Regularization        | Added L2 regularizer to last loss. 0.005      |

The model is implemented in [TF_newKeras](./programs/TF_newKeras.py) and was later translated to raw Tensorflow for the notebook.

### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the mdel I used the Adam Optimizer with a learning rate of 0.0003 and categorical cross entropy as the loss function. During training dropout and L2 regularization were active.

Results of training the raw Tensorflow model and the Keras one are a little different. It is true that in the Keras model I used a callback to save model data and structure only when it maxed the validation accuracy. Results are commented in next section.


### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

This table show the end results of the training:

|                     | Keras Model   | Raw Tensorflow |
|:-------------------:|:-------------:|:--------------:|
| Training Loss       | 0.0310        | 0.000303 (?)   |
| Training Accuracy   | 0.9952        | 0.9999         |
| Validation Loss     | 0.0740        | 0.0737         |
| Validation Accuracy | 0.9990        | 0.9839         |
| Test Loss           | 0.1651        | 0.2370         |
| Test Accuracy       | 0.9724        | 0.9661         |

Discovery of the solution began with raw Tensorflow and developing a small program that implemented a LeNet network with variable number of kernels for each layer. The program accepted a table with the layer sizes, droput value and generated a training run for each one and a .cvs resume to be analyzed later. That allowed to send a batch of work for the night and see the results in the morning a I had not access to a GPU and things where really slow. Each run was stored in a folder and different models saved. An example of the results may be found in [tf_index.csv](./programs/tf_index.csv) and a bit more worked in [tf_resum](./programs/tf_resum.xlsx).

What I found was that it was really easy to get good results in the validation test but results were disappointing in the test set. Also I got quite hooked in the optimization game and have done more than 100 training runs, every one learning somethnig about the network.

One version of the program is [myTrafficSignClassifier_1](./myTrafficSignClassifier_1.py).

Following discussions with my mentor I understood that getting the best epoch of a run was not chating so I implemented the procedure in the program. It uses validation accuracy as the decision variable.

After having the Keras lesson I decidet it would be much easier to work with Keras and will allow me to modify things faster. Mainly adding and removing layers.

Also I read some papers from the NVidia "End to End Learning for Self-Driving Cars" ,
"Optimizing Neural Network Hyperparameters With Gaussian Processes for Dialog Act Classigication" by Frank Dernoncourt and Ji Young Lee, "Image net Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton, "Multi-column depp neural network for traffic sign classification" by Dan Ciresan, Ueli Meier, Jonathan Masci and Jürgen Schmidhuber, "Detection fo Traffic Signs in Real World Images: The German Traffic Sign Detection Benchmark" by Sebastian Houben, Johannes Stallkamp, Jan Salmen, Marc Sclipsing and Christian Igel
between others.

They gave me ideas about the preprocessign  and doing a deeper network.

After some experiments with Keras and, now yes, with a AWS GPU I centered in the final architecture implemented in the [TF_newKeras](./programs/TF_newKeras.py) program. It lacks the batch abilities but is quite useful.

To be able to analyze the trainig results I developed the [ShowSignals_1](./programs/ShowSignals_1.py). It is a quick and dirty program but very flexible modifying comments in the code.

It has 2 functions.
    * Select some random images from the training set ans present their image Processing
    * Compute the predictions and compare with the correct one
    * Take a full image set and compute the predictions, compare with the truth and generate a confusion map in csv and show the erroneous labeled images.
    * Also apply the layer visualization to the first 2 layers

It may read our normal pickled data, read images from a directory which are prefixed by <correct_label>_ name of file and read from the full German Data original files, select the roi from the images and scale them to the 32x32 format, and do the same for the German Data Test. In this case the labels are loaded from our test.

### Image Processing

![Image Processing][image_processing_2]


### Confusion map

Confusion maps give a very good idea about the cnn performance and where it needs improvement. ShowSignals may create the confusion map in cvs format that may be importe to a spreadsheet for anaysis and plotting.

When applied to the test dataset it gives the results in [Confusion Document](./programs/Test_set_summary.xlsx).

An image of the confusion map:

![Confusion map][confusion]

The convergence of the training may be analized from the [TF_newKeras log](./programs/training.xlsx).

![Loss evolution][Loss]
![Accuracy evolution][Accuracy]

A curious observation is that at the beginning results on the validation dataset are better than in the training one. One explanation may be the snall size of the validation dataset and that perhaps only represent the most easily recognised images.


## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Well, I used 8 images just to made two beautiful lines :)


![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

Images were cropped from general ones. The Right of way is not perfectly square which may introduce some recognising problems and the General Caution has a face in it.



### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![Predictions][prediction]

The model predicted correctly the all images except the **General Caution** that has been labeled as **Traffic signals**. That behavior is repeated in other tests and is probably due to the low importance of color in the signal recognition.

I don't think is reasonable to compare the result with these images (just 8 so a 87.5% result) with the one of the full test set. Just one error means a 12.5%.



### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .79         			| Speed Limit (30km/h)  						|
| 1.0       			| Speed Limit (70 km/h) 						|
| .97					| Pedestrians									|
| .88	      			| Traffic Signals (was General Caution)	        |
|                       | General Caution had a 0.002 probability       |
| .63				    | Road work          							|
| 1.0				    | No Entry          							|
| 1.0				    | Trun Right Ahead        						|
| 1.0				    | Right of way at next intersection        		|

So the Cnn is fairly sure of its results but also very sure of its erroneous result.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

I applied the code to a **Speed limit (80 km/h)** signal.

![original][original]

Results from Layer one and two are shown following:

![First Layer][layer_1]
![Second Layer][layer_2]

### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From the layer 1 output it is clear that "something round" is good.

Also seems important a "white" center. Layer 1 is also clearly computing edges.

Layer 2 simplifies a little to show arcs and some "8" or "0" like forms.

Not able to get many conclusions. Perhaps should be better to see how are the weights of the kernels?
