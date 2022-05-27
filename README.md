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
