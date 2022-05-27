# Digit Recognition using Convolutional Neural Networks
Image recognition (digits 0 - 9 from the MNIST dataset) using Sequential and Convolutional Neural Networks (CNN).

## Data
The <b>training data</b> consists of 60000 images of handwritten digits from the MNIST dataset, together with up to 1600 images drawn manually in a tkinter GUI that was developed here, and 42000 images from the Kaggle machine learning competition ["Digit Recognizer"](https://www.kaggle.com/competitions/digit-recognizer/).
The <b>testing data</b> consists of 10000 MNIST images with known labels and 28000 Kaggle images without labels for competition scoring.
<b>Format</b>: images have black background and white-grey digits. Digits are represented as a 28×28 matrix where each cell contains grayscale pixel value from 0 (black) to 255 (white). There are 10 classes, one for each digit.

## Scoring of hyperparameters
Neural networks were scored in three ways:
+ using 10000 MNIST test images,
+ by hand-writing digits in a tkinter Graphical User Interface (GUI),
+ by comparing labels predicted for 28000 Kaggle images with the true labels (undisclosed) that I obtained by reaching the 100% prediction accuracy in the competition.

## Neural Network models
### Sequential NN model
<b>Hyperparameters</b> of the sequential NN model were varied around the following reference values.
| Layer | Neurons | Activation | dropout |
| ----- | ------- | ---------- | ------- |
| Dense | 50 | ReLU | <b>0</b> |
| Dense | <b>50</b> | ReLU | 0 |
| Dense | 10 | softmax | - |

Varied hyperparameters are in bold. The model was <b>compiled</b> with the Adam <b>optimizer</b> and 'categorical_crossentropy' <b>loss function</b>. Reference value of a training parameter: epochs = 200. Minimal number of trials in recognizing the handwritten images was 200, often used value was 500, and the maximal one was 1150.

### Convolutional Neural Networks (CNN)
A CNN model generally consists of convolutional and pooling layers. It works better for data that are represented as grid structures, - this is the reason why CNN works well for image classification problems. One image is distinguishable from another by its spatial structure. Areas close to each other are highly significant for an image.
  The dropout layer is used to deactivate some of the neurons and while training, it reduces overfitting of the model. It randomly kills each neuron in layer of a training set with probability $p$, typically 0.5 (half of the neurons in a layer are dropped during the training). Therefore, the network cannot rely on activation of any set of hidden units, since they may be turned off at any time during training, and the model is forced to learn more general and more robust patterns from the data. We do not use dropout during validation of the data.
  <b>Convolution layers</b> use a <b>kernel</b> (filter, or matrix), usually 3 x 3 or 5 x 5. Center element of the kernel is placed over the source pixel. The source pixel is then replaced with the sum of elementwise products in the kernel and corresponding nearby source pixels. Convolving the kernel over the image in all possible ways gives 2D <b>activation map</b>. Convolving decreases the spatial size.

<b><i>Max Pooling</i>:</b> Keep only a maximal value from each block, e.g., 2 x 2.
<b>Hyperparameters</b> of the sequential CNN model were varied around the following reference values.

| Layer | Neurons | Activation | kernel_size <br>/ pool_size | dropout |
| ----- | ------- | ---------- | ----- | --- |
| Conv2D | 32 | ReLU | 3, 3 | 0 |
| Conv2D | 64 | ReLU | 3, 3 | 0 |
| MaxPooling2D | - | - | 2, 2 | 0.25 |
| Conv2D* | <b>0</b> | ReLU | 3, 3 | 0 |
| MaxPooling2D* | - | - | 2, 2 | 0.25 |
| Flatten | - | - | - | - |
| Dense | 256 | ReLU | - | 0.5 |
| Dense | 10 | softmax | - | - |

*optional layers

The CNN models were <b>compiled</b> with the Adam optimizer as a benchmark. Reference value of a training parameter: epochs = 200.

## User feedback via GUI
User feedback is enabled via a tkinter <b>GUI</b> created here and allowing user to draw a digit using mouse, see its neural network model-generated classification as one of the digits, allow specifying a correct label in case it is misclassified, save the misclassified image and its label for model re-training, and keep track of prediction accuracy. The GUI is implemented using Tkinter library that comes in the Python standard library. The App class is responsible for building the GUI for our app. It has a canvas where one can draw by capturing the mouse event. Functions are triggered by pushing control buttons: button 'Clear' clears canvas and button 'Recognise' activates the function predict_digit() to recognize the digit. This function takes the image as input and then uses the trained model to predict the digit. The predicted label and its probability percentage are displayed.
#### Instructions for GUI:
1. Start by drawing a digit in the canvas window (60 x 60 pixels).
<img src="gui1.png"/>

2. Press “Recognise” and see the predicted class and its probability.
<img src="gui2.png"/>

3. If classified correctly, press “Clear” and go to Step 1.
4. Otherwise, if the handwritten digit is misclassified, press "Fix" button to add the image to train set, which will be concatenated with the MNIST data before next training cycle.
5. Type in correct label and press "Get label" button to add the label to train set.
<img src="gui4_label.png"/>

6. Press "Save corrections" button to save manually labelled images and labels, and clean memory.
7. Current accuracy % of recognizing hand-written digits is displayed on the bottom right.

## Results
The recognition of handwritten digits was tested by applying the model to 10,000 test images and in the actual test by handwriting the digits in the GUI window. We iterated over the digits from 0 to 9 until accuracy curve has reduced its fluctuations. During the development of GUI, 78 misclassified digits were encountered, which were not saved. After implementing the user feedback in the GUI, all misclassified images and their correct labels were saved in two CSV files (all [images](images.csv) in one file and all [labels](labels.csv) in a separate file).

### NN models
The following <b>hyperparameters</b> were optimized:
* $n_{2}$ – number of neurons in layer 2 of NN model,
* dropout,
* epochs,
* $m$ - number of manually labeled images added to training data.

Testing a few values of $n_{2}$ using GUI shows that the optimal number of neurons in layer 2 is between 25 and 75, unless there are several maxima in accuracy.

<img src="accuracyH.nn_50_d0_50_ml0_e200.neurons.png"/>

The effect of dropout after first layer is negative: accuracy decreases for the three tested values dropout = {0.2, 0.5, 0.8}.
<img src="accuracy.nn_50_50_ml0_e200.dropout.png"/>

Longer training of NN (epochs = 400) led to overfitting: accuracy has dropped from 0.694 down to 0.588.
<img src="accuracy.nn_50_d0_50_ml0_e200_400.epochs.png"/>

<img src="?"/>
