# Q-AE
An autoencoder based on new types of neurons
There are 5 quadratic convolutional layers and 5 quadratic deconvolutional layers in Q-AE, 
where each layer has 15 quadratic filters of 5×5, followed by a ReLU layer. Zero paddings are used in the first four layer, 
therefore the fifth layer is the bottleneck layer. 

The core part of this code is to define "quadratic convolution":

tf.nn.relu((conv2d_valid(input, W_r)+b_r)*(conv2d_valid(input, W_g)+b_g)+conv2d_valid(input*input, W_b)+c) 


