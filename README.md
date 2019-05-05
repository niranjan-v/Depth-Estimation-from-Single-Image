# Depth-Estimation-from-Single-Image
Depth Estimation from Single Image using CNN, CNN+FC, CNN-Residual network

## Objective
Given a single image we have to estimate its depth map.

## Solution Approaches
### METHOD 1  -  CNN+FC Network:
The network consists of 4 convolutional layers followed by two fully connected layers. The number of parameters in the fully connected layers are 233989700 which is much higher the number of parameters in convolutional layers which is 51040. Adam optimizer is used with learning 1e-3 and weight decay 1e-4. The presence of fully connected layers allows overfitting the model on training set but fails to provide reasonable performance on test set 

## Dataset
We have used [NTU-RGBD Action](https://github.com/shahroudy/NTURGB-D) dataset in this project.
It consists of 60 classes of various Human Activities and consist of 56,880 action samples. Of these 60 classes we removed the last 11 classes consisting of multiple people. 
We trained most our models on subsets of this dataset consisting of
