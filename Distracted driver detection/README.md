# Distracted driver detection

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/81b6b339-f6a0-4c18-8b25-3b8a97bd0a94)

## Introduction
According to the CDC motor vehicle safety division, one in five accidents is caused by a distracted driver. Sadly, this translates to 425,000 people injured and 3,000 people killed by distracted driving every year. 

Shocked by the statistics, this project was developed based on the Kaggle challenge originally brought up by an insurance company to see whether a dashboard camera can capture the distracted driver and elaborated further on adding data augmentation and application to real-world scenarios. 

## Methods
### Data Processing
There are 3 kinds of differnet processing:
1. Without data augmentation (applied to all models): CenterCrop + Normalize
2. With data augmentation (applied to VGG16): RandomRotation + ColorJitter + RandomResizedCrop + RandomHorizontalFlip + Normalize
3. Grayscale downsize (applied to baseline CNN): Grayscale + Resize + Normalize.

The first 2 kinds are for the purposes of comparing the performance of dat augmenation. The last kind is for better performance when inputting to the CNN model.

### Models
1. VGG16
   
VGG16 architecture is composed of a stack of 16 convolutional layers, where filters with a small kernel size (3x3) are used, and 3 fully-connected layers following the stack of convolutions.
We select VGG16 because it is one of the best performing architectures and has achieved a top-5 error rate of 7.3% (92.7% accuracy) in ILSVRC — 2014

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/a28660d9-5950-40eb-addd-e84c887c927a)

2. ResNet18
   
   ResNet18 is an 18-layer-deep CNN that belongs to the ResNet family. The highlight of its architecture is skip connections that can mitigate exploding or vanishing gradients.

3. AlexNet
 
We are not new to AlexNet, as it was introduced in the lecture and was implemented in Lab2. AlexNet is well-known as the winner of the 2012 ImageNet LSVRC-2012 competition. The highlights of its architecture include [Ref.5],

Using ReLU instead of Tanh for activation considerably enhances efficiency.
Using dropout instead of regularization to mitigate over-fitting.
Adding pooling layers to reduce the network size.
The input for AlexNet should be an RGB image. In the above code, the images have been cropped to 224x224 to be suitable to be fed to AlexNet. AlexNet has more than 50 million trainable parameters and 650,000 neurons. Also, we know that AlexNet consists of 5 convolutional layers and 3 fully-connected layers. Its detailed architecture is depicted in the following figure.

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/7663251a-5ae1-4abb-9c74-63fb7ec55022)

4. (3-layer Baseline CNN)
   
Now, we will design a 3-layers CNN model as our baseline model in which each convolutional layer consists of one convolution that doubles the channels and a "same convolution" that has identical input and output channels. Then it's followed by 3 fully-connected layers where the first two layers are refined by relu activation. And the last fully-connected layer outputs 10 classes.

The model architecture can be easily engineered to be compatible with various-sized inputs, including large-size RGB images (3x224x224), and downsized grayscale images with 3 output channels (3x64x64), downsized grayscale images with 1 output channel (1x64x64), aligning with the transformations that are defined in section 1.1.


### Testing

In this part, we would like to compare the performance of all the models we have. Two Numerical metrics, test accuracy and confusion matrix will be used as quantitative analysis. The test accuracy tells the models' overall performance whereas the confusion matrices show the predictions vs. true labels per image class. We will also use Gradient-weighted Class Activation Mapping (Grad-CAM), which is one of the explainable AI techniques that can visualize feature importance to the output on the original images, to analyze the results qualitatively.

We summarize the test accuracies in a bar plot as follows, which reflects that except for the VGG16 model trained with augmented images, all the rest four models achieve impressive test accuracies. The VGG16 model shows much inferior performance because a large amount of information has been lost when random cropping or rotation is applied to the training set. The reason why the ResNet18 model has the second-lowest test accuracy is that it has the fewest trainable parameters, which might be insufficient, when all its convolutional layers are frozen.

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/0a32b49e-da7c-4c99-93c7-426290469870)


### Application
To be completed.
## Findings
To be completed.
