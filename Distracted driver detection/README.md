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


### Evaluation

In this part, we would like to compare the performance of all the models we have. Two Numerical metrics, test accuracy and confusion matrix will be used as quantitative analysis. The test accuracy tells the models' overall performance whereas the confusion matrices show the predictions vs. true labels per image class. We will also use Gradient-weighted Class Activation Mapping (Grad-CAM), which is one of the explainable AI techniques that can visualize feature importance to the output on the original images, to analyze the results qualitatively.

We summarize the test accuracies in a bar plot as follows, which reflects that except for the VGG16 model trained with augmented images, all the rest four models achieve impressive test accuracies. The VGG16 model shows much inferior performance because a large amount of information has been lost when random cropping or rotation is applied to the training set. The reason why the ResNet18 model has the second-lowest test accuracy is that it has the fewest trainable parameters, which might be insufficient, when all its convolutional layers are frozen.

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/0a32b49e-da7c-4c99-93c7-426290469870)

In the following sections, we will look into the decision-making of our models using Grad-CAM. Grad-CAM uses the class-specific gradient information flowing into the final convolutional layer of a CNN to produce a coarse localization map of the important regions in the image . Grad-CAM features its applicability to a wide variety of CNN models, making it possible to be applied to all the models we have built as long as the final convolutional layer is specified.

The resulting heatmaps highlight the important features are generated by scoring the feature maps in the last convolutional layer with the "average gradients", which are calculated by backpropagating the output signals.

_True label: C2_

_Predicted label: 2_

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/3c055697-143d-4a6c-9c0d-c71efdc1d28e)

_True label: C3_

_Predicted label: 3_

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/f497e95e-bcad-4b4c-9362-d170a2113ceb)

_True label: C6_

_Predicted label: 6_

![image](https://github.com/pengp5781/Deep-Learning/assets/111671117/320c0842-48f9-481b-8745-a858b74cbdf0)

To demonstrate our trained model, we first used the images from the "test" folder on Kaggle. Those images were unlabeled and had never been exposed to the training pipeline. To make it easier for visualization and model evaluation, we processed each image and embedded it with a list of labels with probabilities.

We tried all the models we had trained and it turned out the **VGG16 model trained with data augmentation** yielded the most accurate predictions. Thus, we picked this model for the demonstration. Firstly, we resized the images to the same dimension as we used while training. Secondly, the same normalization was then applied. OpenCV library was used to draw the embedded prediction results and video creations.

### Application

To further evaluate our models, we applied them to a real-world video that was taken through a dashcam mounted in the car. This was intended to test the model performance in real-world scenarios and to see if the model could be applied to live driving, which exactly aligned with our motivation to pick this topic, with image streams and formats different from the training data. The video was read using the OpenCV library and every single frame was passed through the same pre-defined transformations as we did in training.

The highlights in the demonstrations include:

* Robust distracted behavior detection on still images.
* Low-latency processing (prediction time < 10ms) makes real-time prediction and streaming possible.

The demonstration on the real-world video reflects a bunch of challenges in making correct predictions:

* Requires a fixed camera angle, making it less compatible with different viewpoints or scenarios.
* The classifications of specific classes (drinking vs. talking to the phone) need to be improved.
* More likely to make a false prediction when there is a movement and those false predictions typically lie in a motion category. Such as reaching to the back or touching the hair.

## Findings

In conclusion, we find out that both the 3-layer baseline CNN and VGG16(no augmentation) are the most competitive models for our specific task with AlexNet following closely. Leveraging heatmap-based explainable AI, we realize that different models tend to focus on varying regions of interest and emphasize different pixels. The distinction is significant and can be utilized to select a model when we seek to prioritize exploring a specific area.

Data pre-processing has played a crucial role in our project. Originally it is only applied to match the dimensions of inputs to the transfer learning models. However, after investigating the cropped image, we realize that cropping does more than that -- it gives a more concentrated image that contains just sufficient information to make a good prediction. In addition to that, when facing a similar task, one might also want to consider the option to resize the inputs to reduced resolutions and transform them into different color spaces as well for accelerating training. In our case, 64x64, grayscale images are fed to the baseline CNN and it is far from compromising the performance -- as it yields promising performance just as the other more complicated models with higher-resolution inputs. But of course, this is case-dependent so one needs to carefully investigate the data and task to see if such a method fits.

As to explain the overall descent performance for all models, we think it is due to there being plenty of training data available and the pattern is easier to locate in our task. To be more specific, since all images are captured by dashcams, they are all from identical viewpoints which greatly reduces barriers to training. Plus, the object (human) is always located in the center and the head and hands that the models look for are normally placed in a specific area. Moreover, the driver's cabin is a rather simple and consistent background. These factors all together create an ideal environment for learning and simplify the task.

However, one downside associated with our method is that the models do not perform, or in other words, generalize well on newly captured live data, which is mainly attributable to the different viewing angles, lighting conditions, and the more complicated transition between movements that might not even belong to one of our categories. Augmenting the raw images has proved to be effective in our implementation as the randomness in the transformations adds robustness to the models by making the models more adaptable to a variety of images. The solution to this can be: 1) including more photos from different viewing angles if possible or introducing noise to current photos to make the model robust just like the augmentation we have performed but with more precision. 2) As the model might have some trouble classifying specific distracted behaviors, it is still good at detecting whether there is a distracted behavior. We could simply reduce its capability to make only binary classification which still fits the general requirement for detecting distracted drivers!
