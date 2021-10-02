# PROJECT TITLE: CLEAN/DIRTY ROAD DETECTION USING TRANSFER LEARNING


# Description:
This is a project based on ConvNets used to identify whether a road is clean or dirty. We have used <em>MobileNet</em> as our base architecture and the weights are based on <em>imagenet</em>.

![Alt text](https://github.com/FaizalKarim280280/Clean-Dirty-Road-Classifier/blob/main/Plots/prediction_result.png)

# Contents
  1. Dataset download.
  2. Tools and Libraries.
  3. Code.
  4. References.

## 1. Dataset downlaod
We have used web scraping techniques to download 238 images of both clean and dirty roads. This dataset was completely used for training the model.

## 2. Tools and Libraries
![Alt text](https://github.com/FaizalKarim280280/Clean-Dirty-Road-Classifier/blob/main/Plots/logo.jpeg)<br><br>
In this project we will be using the ***Keras Deep Learning*** Library and we will be running it on top of the Tensorflow backend.\
Others include:
  * Numpy 
  * Matplotlib
  * OpenCV

## 3. Code
<em>ClassifierModel.py</em> is the main code which encapsulates the CNN architecture that was used for this project.
  * We have used <em>keras.applications.mobilenet_v2</em> to import the MobileNetV2 architecture for our base model, with <em>training = False.</em>
  * On top of it, we have added two dense layers consisting of 512 and 256 units both having ReLU activation followed by Dropout.
  * The output layer is a simple 2 unit softmax layer.<br>

### Training Accuracy and Loss 
![acc](https://user-images.githubusercontent.com/79277882/135721088-ff910033-c625-435e-8a61-ed57e685e3fa.png)
![loss](https://user-images.githubusercontent.com/79277882/135721115-5ba27aef-fbae-4b35-b0fa-5a3c0ad9ff8c.png)

We have used <em>keras.losses.CategoricalCrossentropy</em> as our loss function, <em>keras.optimizers.Adam</em> as Optimizer and trained the model for 15 epochs. 
## 4. References
  * [Tensorflow](https://www.tensorflow.org/api_docs/python/tf)
  * [Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
  * [Web Scraping](https://realpython.com/python-web-scraping-practical-introduction/)
  * [MobileNet](https://arxiv.org/abs/1704.04861)

