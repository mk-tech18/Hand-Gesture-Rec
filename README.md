# Computer Vision -- Hand Gesture Recognition:
>  Author: K_M
>  Nov. 2022

---
## Introduction
### Project Idea
The main project goal was to build up a model that could recognize hand gestures and have some interaction on the neato side.

### Learning Goal
I wanted to learn more about machine learning, especially computer vision. Therefore this project was more focused on the application of computer vision rather than robot interaction like in previous projects. Computer vision neural networks are large and hard to train because of the need for a huge database. Therefore I utilized transfer learning with various available pretained models and different preprocessing images.

## Data Preparation
I used the onboard camera from neato and the webcam from my laptop with openCV to record videos with different hand gestures and save the frames as the training, validation, and test datasets.
### Image Recording
I recorded a series of images from the webcam with 4 poses: `['come', 'left', 'right', 'stop']`, each with approximately 1000 images. 
Photo taken from webcam:

Photo taken from neato:

![Neato images](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/neato.png)

The webcam set works better due to resolution and lighting conditions.
### Image Preprocessing
I tried various ways for image preprocessing, some of them help and some do not.
#### Data Augmentation
The most crucial one was to use data augmentation to reduce overfitting and increase sample diversity for this small-size dataset. I added random rotation and vertical flip to the dataset. (There was no horizontal flip as standard data augmentation because it would confuse the left/right category).
#### Train with Original Dataset
The original dataset did not work well with overfitting (Extremely high accuracy in training but poor test accuracy.) because there were too many distractions and noises in the background.  Therefore, I tried several ways to reduce noise or only keep important information.
#### Edge Detection
I tried to use edge detection to move away the distraction from colors and background. Even though the resulting image looks good before compression, this does not work well because the resolution is even worse after reshaping and the "single line" edges:

![Edge Detection](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/original_edge.png)

#### Binary Filter
The technique with binary image was to remove everything else but pixels with my skin-like color. This significantly reduces unnecessary information but the potential problem is consistency with users of different skin tones or even a similar background color could confuse the model.

 ![Binary Filter](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/binary.png)
 
#### Edge Detection  with Binary Image
I then tried combining edge detection on the binary image.
It works slightly better than applying edge detection to the original image, but not as well as the binary image

![Edge Detection on Binary Image](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/binary_edge.png)


## Model with Transfer Learning
I used pre-trained models provided by TensorFlow as the base model for transfer learning. 
### How transfer learning works
We use a pre-trained model with classification ability and make it learn to classify images of our own classes. Transfer learning is widely used in the field of computer vision (especially image classification), natural language processing, and speech recognition. They could boost performance for a small dataset with specific tasks.

### Base Model
I tried using MobileNet_v3, inception_v3, and inception_resnet_v2 as base models and inception_resnet_v2 produced the best result using the same datasets. This might be because this model has the largest network. 

#### Feature Extraction
I started with a frozen base model without its output layer and added a few output layers: global average pooling to convert the features of each image to one single column vector, a random dropout layer to prevent overfitting, and a dense layer to produce a prediction vector for each image (possibilities for 4 classes, and we take the class with the highest score as the predicted class). We could see after 10 training epochs, both the training and validation accuracy improved, which means that the output layer is learning to classify the images:

![Training Output Layer Only](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/initial_outcome.png)


#### Fine Tuning
To further improve the model performance, we could modify the base model with limited variations to not mess up the whole pre-trained model:
We first started by only training the first 100 layers of the base model with 130 trainable variables. To prevent exploding gradients, we decrease the learning rate by a factor of 10.
The performance significantly improved with fine-tuning.
**Decision**: How many layers to train?
In model training, I noticed that too few layers (50 out of 300 layers with around 50 trainable variables) would make the training time much quicker but with lower accuracy. However, too many trainable layers (200 layers with 150+ trainable variables) significantly slows down the training with bad output because we will be overfitting (high training accuracy) the whole neural network by changing so many outputs. As a result, keeping 
both the trainable layer number and variables around 100 yields a balanced result.


### More on Inception_resnet_v2
Inception_resnet_v2 is a newer model published in 2017 based on inception architecture with integrated residual connections. Therefore it combines multiple-sized convolutional filters with residual connections which could avoid the degradation problem (The problem that deeper networks should allow a higher level of feature learning but instead causes more error.) and allow lower training time. 

![Overview of Inception_Resnet_v2](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/inception_resnet_view.png )

### Model Outcome
Here is a video demo of the real-time prediction of the model:

<a href="url"><img src="https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/demo.gif" width="2000" ></a>

We could see that the model predicts the outcome pretty well with some small fluctuations which would be filtered out if continuous detection is required to trigger the response.


## Challenges
### 1. Environment Setup
Google colab was not fast and stable enough for thousands of images. However, setting up the GPU environment on an intel mac could be painful. Luckily I will not have to do this again in the future.
### 2. Constant Zero Loss
At the first several trainings, I kept getting zero loss from both training and validation with ~60% accuracy. It took me a tremendous amount of time from useless training and troubleshooting. From learning through parameters of `image_dataset_from directory` and different types of loss functions, I fixed this issue by setting `label_mode` to `int` and using sparse categorical cross-entropy. 


## Key Takeaways
### 1. Transfer Learning
I was able to use TensorFlow pre-trained model APIs for transfer learning which was really exciting and I realized they are extremely useful in computer vision because they could perform really well even if they were not for some specific tasks.

### 2. Tuning in Neural Networks
Even though the model training process was time-consuming and a little boring, I was able to find many interesting facts that would be useful in my future machine-learning projects. We already talked about the selection of the number of trainable variables which was specific to transfer learning, here are some more general ideas in machine learning.
1. Low dropout could lead to overfitting. I started with a 0.2 dropout rate in the prediction layer and it led to obvious overfitting as the training loss was so low and validation loss not really improving over epochs:
2. 
 ![dropout = 0.2](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/low_dropout.png)
 
 Therefore I turned to a 0.4 dropout rate which looks a little scary because we would be just cutting off 40% of the connections right before the final output. But it turned out much better:
 
 ![dropout = 0.4](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/fin_tuning.png)
 
2. Epoch selection: I thought more epochs would make the neural network learn more features. But it turns out too many epochs could be not only time-consuming but also leads to overfitting. On a small dataset, around 20 epochs would be enough and we could utilize a callback function called `EarlyStopping` where we could stop the training before reaching set epochs if we do not see significant improvement over some period of training time. 
3. Learning rate: Too low of a learning rate might make the training extremely slow and get stuck but making them too high could make us miss the optimal solution because the steps are too big.
4. Loss functions: 
	Loss functions are how we measure how far is our prediction to the correct output. Our weights for each neuron are determined through the process of lowering loss. There are many types of loss functions and they each serve different modeling decisions. I was only able to 	earn about the most used types:
	- Mean Squared Error: This is most used in regression problems where we need the output to be a continuous value.
	- Binary Crossentropy: 
		$\text{Loss} = -\frac{1}{N} \sum^{N}_{i = 1} y_{true} \log{(y_{pred})} + (1-y_{true}) \log(1-y_{pred})$ 
	This is the most used in yes or no, or binary classification questions.
	-  Categorical Crossentropy:
		$\text{Loss} = -\frac{1}{N}\sum^{N}_{i = 1} y_{true} \log{(y_{pred})}$
		This is used in multiple-label classification questions.

5. Activation functions:
	Activation functions are used to produce one final result as output from the feature vectors. It is also used in hidden layers 
	- **Sigmoid**:  $f(x) = \frac{1}{1+e^{-z}}$
		- This is used in the output layer **binary classification** (must) and **regression problems**, with the output value from range 0 to 1.
		- It is also used in hidden layers of RNN, together with tanh.
		-  Drawbacks: 
			- It is centered at 0.5 ($f(0) = 0.5$), making the optimization process computationally harder.
			- It has the vanishing gradient problem. (See ReLu part.)
	- **Softmax**: This is used in **multiple label classification** because it calculates the probability of N different events with their sum equal to 1. This would make sure all class possibilities are mutually exclusive.
	- **ReLU** (Rectified Linear Unit): $f(x) = \text{max}(0,x)$
		- It is the most used activation function in **hidden layers** of MLP and CNN models because it is not only fast but also solves the vanishing gradient problem. 
		- ReLU has a faster convergence speed because it has no exponential terms and a fixed derivative.
		- Drawbacks:
			- Dying ReLu problem because of the 0 value.
## Lessons Learned and Future Improvements
### Model vs. Ros Implementation
Even though my main focus was computer vision model from the very beginning, I still wanted some implementation with ros because of the nature of this class. However, making changes in machine learning is not like other types of programming where you could see the result of the change immediately after you make it. It takes almost 1 hour to train a single model (much much longer in google colab) and during that time I could barely do any other things because the laptop might just crash.  I'm glad I was still able to take some basic interaction about vision from neato by taking photos from it. 

In the future, if I were to implement some computer vision project, I would get a remote GPU server for everything to run more quickly and without sacrificing my time to the intense computation.  There are so many things that could be done with a moving robot that can see.

### Object Tracking
I wanted to achieve the pose tracking algorithm as in mediapipe: 

![mediapipe hand](https://raw.githubusercontent.com/AlexisWu-01/compRobo22_computer_vision/main/demo/handpose_demo.png)

However, I could not find a pre-trained model for such pose detections. If there were more time I would try to build and train an object-tracking machine-learning model. 


## Deep learning could be not as fancy as it sounds (Tuning vs Creating Model)
I wanted to learn more about the theories and math behind the neural networks but could not realize this with transfer learning even though it provides a much better result in an engineering outcome aspect. The main working flow was finding some APIs, then looking into their codes and function, learning about different parameters, and seeing their impact on the output. Waiting for an outcome can be boring but I was still glad to learn so much about OpenCV implementations and building up models from TensorFlow.
