**Image Classification**

Image classification is the process of assigning classes to images. This is done by finding similar features in images belonging to different classes and using them to identify and label images. Image classification is done with the help of deep learning/neural networks.

**About the dataset**

ImageNet-100
ImageNet-100 is a subset of ImageNet-1k Dataset from ImageNet Large Scale Visual Recognition Challenge 2012. It contains random 100 classes as specified in Labels.json file.
1. Train(train) Contains 1300 images for each class.
2. Validation(val) contains 50 images for each class

**Downloading the dataset**

To download the dataset, follow the steps mentioned in the 'imagenet.ipynb' and run the first 4 cells.

=======================================================

Note: You must download the API token from kaggle. Follow the below steps to download the API token.

1. Login to your kaggle account
2. Click your profile icon
3. Go to settings
4. Scroll down to API, you will be able to see 'create new API token'.
5. Click it, kaggle.json file will get downloaded
   
========================================================

**Installing the dependencies**

Create a virtual environment if required, using the below code:
--> python -m venv vnv
--> source vnv/bin/activate

Install the requirements, as follows:
pip install tensorflow
pip install tqdm
pip install numpy
pip install scikit-learn
pip install keras
pip install pandas
pip install opencv-python
pip install shap
pip install matplotlib
pip install -U efficient

The complete code for installation can be found in imagenet.ipynb file itself

**Dataset preparation**

We find 4 different training folders, with each containing 25 folders specifing 25 classes. These, folders are merged together to have a single train folder and val folder.
Follow the same code as specified in imagenet.ipynb file

**Prepocessing and Training the dataset**

Follow the code as specified in imagenet.ipynb.

**Preprocessing**:

Keras ImageDataGenerator is used in the realm of real-time data augmentation to generate batches comprising data from tensor pictures. We may utilize the ImageDataGenerator resize class by supplying it with the proper parameters and the relevant input

===========================================================

Note: 
1. batch_size is set as 32 and can be varied
2. class_mode is set to 'categorical', since we have 100 classes in total. When only 2 classes are used, it can be changed to 'binary'
3. target_size can also be varied

===========================================================

We will be using only the basic models, with changes made only to the final layer. This is because this is just a 100 classes classification problem while these models are built to handle up to 1000 classes.

The 'choose_model' variable can be changed to utilize any one of the below mentioned architecture to train the dataset.
1. VGG16
2. Inceptionv3
3. ResNet50
4. efficientNet
Here, I have set choose_model='vgg16', as I am using VGG16 architechture for training the model.

Since we don’t have to train all the layers, we make them non_trainable
--> for layer in base_model.layers:
-->     layer.trainable = False

We will then build the last fully-connected layer. I have just used the basic settings, but feel free to experiment with different values of dropout, and different Optimisers, activation functions and learning rate.
1. Dropout is set to 0.5. This can be changed to experiment the model's performance
2. Activation function used is 'relu', which can be replaced with other functions. For example: 'leaky_relu'
3. Optimizers used is Adam, which can be changes to RMSprop or SGD
4. Learning rate is set to 0.001. Which can also be changed
   
**Training:**

Early stopping is used. In the simplest case, training is stopped as soon as the performance on the validation dataset decreases as compared to the performance on the validation dataset at the prior training epoch. For example, when loss is increasing in validation set
Early stopping is a form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent. Such methods update the learner so as to make it better fit the training data with each iteration
It optional to use early stopping.

Here, the models are trained only for 3 epochs. The no.of epochs can be increased to 25/50, to obtain good accuracy.
The models are then saved using, model.save('name.h5')

**Evaluation**

1. Graph to check loss and accuracy
2. Classification report
3. Confusion matrix
The prediction of the model for image input is also given.
1. Load the model
2. Input the image path
   
**SHAP value generation**

SHAP (SHapley Additive exPlanations) values are a way to explain the output of any machine learning model. It uses a game theoretic approach that measures each player's contribution to the final outcome. Shapley values are a widely used approach from cooperative game theory that come with desirable properties. It represents how a feature influences the prediction of a single row relative to the other features in that row and to the average outcome in the dataset. The value has both direction and magnitude, but for model training, SHAP importance is represented in absolute value form.
The code to generate shap values can also be found in imagenet.ipynb file

**Post Model Quantization**

Model quantization is vital for deploying large AI models on resource-constrained devices. Quantization levels, like 8-bit or 16-bit, reduce model size and improve efficiency. Model quantization is vital when it comes to developing and deploying AI models on edge devices that have low power, memory, and computing. It adds the intelligence to IoT eco-system smoothly
The code to implement post model quantization can also be found in imagenet.ipynb file.


**Conclusion**

The primary goal is to compare the performance of the imagenet dataset for various architectures. The below architectures are used for the comparative study:
1. VGG16
2. Inceptionv3
3. ResNet50
4. efficientNet
We have used 25 classes with 1300 images each in the training data and 25 classes with 50 images each in the validation data. We have fixed the fully-connected layer and certain parameters to train the model using above mentioned architectures.

Layers defined in the fully connected layers are as follows:

The flatten layer serves the purpose of reshaping the output of the preceding layer into a one-dimensional vector, which can then be fed into subsequent fully connected layers. 
A dense layer is a layer where each neuron is connected to every neuron in the previous layer. In other words, the output of each neuron in a dense layer is computed as a weighted sum of the inputs from all the neurons in the previous layer.
The Dropout Layer. Another typical characteristic of CNNs is a Dropout layer. The Dropout layer is a mask that nullifies the contribution of some neurons towards the next layer and leaves unmodified all others. The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
Finally the output layer

Parameters specified:

1. Dropout is set to 0.5.
2. Activation function used is 'relu'.
3. Optimizers used is Adam.
4. Learning rate is set to 0.001.
5. target_size=(128,128)
6. Epochs = 3

A comparative study is done with the specified parameters.

1. Under 3 epochs ResNet achieved training accuracy of 0.6599 and validation accuracy of 0.7008
2. Under 3 epochs VGG16 achieved training accuracy of 0.6369 and validation accuracy of 0.6584
3. Under 3 epochs EfficientNet achieved training accuracy of 0.0368 and validation accuracy of 0.0400
4. Under 3 epochs Inceptionv3 achieved training accuracy of 0.0364 and validation accuracy of 0.0400

Therefore, ResNet and VGG16 shows good performance under 3 epochs and Efficientnet and Inceptionv3 shows poor performance under 3 epochs and above specified parameters.

The specified parameters can be changed to experiment the results. We have also plootted shap values to see how features contributed for the cllassification problem. Model Quantization is also performed to deploy large AI models by reducing its size and improving efficiency.

========================================================

Note:
Two .ipynb files are uploaded:
1. imagenet.ipynb --> Complete code implementation
2. vgg16.ipynb --> Complete code implementation using VGG16, also by customly changing the choose_model variable, we can implement the same code using Resnet50, inception and efficient architectures too.

========================================================

**Further Optimization**

We see that inception is giving a pretty low accuracy, compared to any other algorithms. To find out the best parameter to train using inception, I have used hyperparameter tuning. The code for which can be found in 'inception_hp.ipynb'

=======================================================

Note: I tried hyperparameter search for :
1. Dense layer unit - 512 and 1024
2. Activation function - 'Relu' and 'tanh'
3. Dropout layer - 0.2 and 0.5
4. Optimizer - Adam, SGD, RMSProp
5. Learning rate - e^-2 and e^-4

You can change these values, add more values or try it for various other factors. This is just a sample code.

========================================================

After performing Randomsearch using keras_tuner, The hyperparameters, to achieve good accuracy using inception architecuture is given in the below:

1. The units of the dense layer in fully connected layer is set to 1024
2. Activation function in the dense layer is set to 'Relu'
3. The drop-out layer is set to 0.2
4. Optimizer is set to 'Adam'
5. learning rate is set to 'e^-4'

For inception architecture, setting up the above parameters in the fully connected layer. We were able to see an improvement in the accuracy. 
The results under 2 epochs are as shown below:

1. Accuracy increased from 20.2% to 37.6%
2. Loss decreased from 2.6 to 2.0
3. Validation accuracy increased from 23% to 26%
4. Validation loss dropped from 2.54 to 2.51

Randomsearch hyperparameter tuning can be implemented for other architectures too, by just changing the choose_model variable to any one of the following, such as vgg16, inception, efficientnet and resnet50. Based on the best hyperparameters obtained, the model can be trained to acheive a better accuracy rate.
The complete code implementation can be found in inception_hp.ipynb.

