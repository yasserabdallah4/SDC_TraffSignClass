## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./Writeup/train_dataset_hist.png "Train Count Visualization"
[image2]: ./Writeup/test_dataset_hist.png "Test Count Visualization"
[image3]: ./Writeup/valid_dataset_hist.png "Valid Count Visualization"
[image4]: ./Writeup/training_data_set.png "Training Data Set"

[image5]: ./Writeup/1.png "Traffic Sign 1"
[image6]: ./Writeup/2.png "Traffic Sign 2"
[image7]: ./Writeup/3.png "Traffic Sign 3"
[image8]: ./Writeup/4.png "Traffic Sign 4"
[image9]: ./Writeup/5.png "Traffic Sign 5"

[image10]: ./Writeup/image1_softmax.png "Image 1 softmax prediction"
[image11]: ./Writeup/image2_softmax.png "Image 2 softmax prediction"
[image12]: ./Writeup/image3_softmax.png "Image 3 softmax prediction"
[image13]: ./Writeup/image4_softmax.png "Image 4 softmax prediction"
[image14]: ./Writeup/image5_softmax.png "Image 5 softmax prediction"

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

#### Submission Files

 1. Files submitted, from everything I read a HTML file, notebook, and write up file is required. So this will meet requirements.
 
 ### Data Set Exploration
 
 ####  The submission includes a basic summary of the data set.
 
 The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the Python's numpy library to calculate summary statistics of the traffic signs data set:
 
 * The size of training set is 34799
 * The size of test set is 12630
 * The shape of a traffic sign image is (32, 32, 3)
 * The number of unique classes/labels in the data set is 43

#### The submission includes an exploratory visualization on the dataset.

The 4th code cell contains the bar graph for "No. of examples for each label" for training, validation and testing dataset.

![alt text][image1]

![alt text][image2]

![alt text][image3]

The 5th code cell contains the image for each labe/class that we have in training data set 

![alt text][image4]

### Design and Test a Model Architecture

#### The submission describes the preprocessing techniques used and why these techniques were chosen.

The 6th code cell of the IPython notebook contains the code for preporcessing which includes:

**Grayscaling the images**, to convert from 3 channel RGB to single channel gray image.

**Normalization of image** data between -1 and 1 instead of 9 to 255 

#### The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

The code for my final model is located in the 7th, 8th and 9th code cells of the IPython notebook.

My final model is based on LeNet architecture and have the following layers:

| Layer                 |     Description                                |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 grayscale image                               |
| Convolution 5x5         | 2x2 stride, valid padding, outputs 28x28x6     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 14x14x6                 |
| Convolution 5x5        | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 5x5x16                 |
| Convolution 1x1        | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU                    |                                                |
| Fully connected        | input 412, output 122                                            |
| RELU                    |                                                |
| Dropout                | 50% keep                                            |
| Fully connected        | input 122, output 84                                            |
| RELU                    |                                                |
| Dropout                | 50% keep                                            |
| Fully connected        | input 84, output 43                                            |


#### The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

The code for training the modelo is located in the 10th code cell

To train the model, I used an LeNet for the most part that was given, but I did add an additional convolution without a max pooling layer after it like in the udacity lesson.

**Optimizer**: AdamOptimizer
**Batch Size** = 156
**Epochs** 27 
**Hyperparameter** - mu = 0, sigma = 0.1, learning rate: 0.00097, dropout rate = 0.5

#### The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 95.0%
* test set accuracy of 92.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
*  I used a very similar LeNet architecture.  I used it because they got such a good score the answer was given through it.

### Test a Model on New Images

#### The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]

I used bad qualities images, even so that classifier gives god results.


#### The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set

#### The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

The code for making predictions on my final model is located in the last cell of the iPython notebook.

Here are the results of the prediction:
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

