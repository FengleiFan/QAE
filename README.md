# Q-AE
An autoencoder based on new types of neurons
There are 5 quadratic convolutional layers and 5 quadratic deconvolutional layers in Q-AE, 
where each layer has 15 quadratic filters of 5×5, followed by a ReLU layer. Zero paddings are used in the first four layer, 
therefore the fifth layer is the bottleneck layer. 

We compare the validation loss with other CT denoising model
![Validation Loss](https://github.com/FengleiFan/QAE/blob/master/Figure%203.tif)
