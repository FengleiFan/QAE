# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:12:08 2019

@author: Fenglei Fan
"""

import numpy as np
import tensorflow as tf

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

import h5py
import os
#import pydicom as dicom
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Fixed random seed
np.random.seed(seed=1)



def read_h5(file_name):
    '''
    read full-size images
    data shape is N x 512 x 512
    '''
    f = h5py.File(file_name, 'r')
    data, label = np.array(f['data']),np.array(f['label'])
    f.close()
    return data, label

# full-size images

def normalize_y(x, lower = -300.0, upper = 300.0):
    x = np.squeeze(x) # remove depth
    x = (x - 1024.0 - lower) / (upper - lower)
    x[x<0.0] = 0.0
    x[x>1.0] = 1.0
    x = np.expand_dims(x,3)
    return x


def bias_variable(shape):
    initial = tf.constant(-0.02, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d_valid(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_same(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d_valid(x,W,outputshape):
    return tf.nn.conv2d_transpose(x,W,output_shape=outputshape,strides= [1,1,1,1], padding = 'VALID')

def deconv2d_same(x,W,outputshape):
    return tf.nn.conv2d_transpose(x,W,output_shape=outputshape,strides= [1,1,1,1], padding = 'SAME')

def Quad_deconv_layer_valid_linear(input, shape, outputshape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(0.04, shape=[shape[2]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[2]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    return (deconv2d_valid(input, W_r, outputshape)+b_r)*(deconv2d_valid(input, W_g,outputshape)+b_g)+deconv2d_valid(input*input, W_b,outputshape)+c

def Quad_deconv_layer_same_linear(input, shape, outputshape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(0.04, shape=[shape[2]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[2]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    return (deconv2d_same(input, W_r, outputshape)+b_r)*(deconv2d_same(input, W_g,outputshape)+b_g)+deconv2d_same(input*input, W_b,outputshape)+c

def Quad_deconv_layer_same(input, shape, outputshape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(0.04, shape=[shape[2]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[2]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[2]],dtype=tf.float32))
    return tf.nn.relu((deconv2d_same(input, W_r, outputshape)+b_r)*(deconv2d_same(input, W_g,outputshape)+b_g)+deconv2d_same(input*input, W_b,outputshape)+c)



def weight_variable_Wr(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)

def weight_variable_Wg(shape):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)

def weight_variable_Wb(shape):
    initial = tf.truncated_normal(shape, stddev=0.01,dtype=tf.float32)
    return tf.Variable(initial)


def Quad_conv_layer_valid(input, shape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))  # W_b can also be initialized by Gaussian with samll variance
    b_r = tf.Variable(tf.constant(-0.04, shape=[shape[3]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[3]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[3]],dtype=tf.float32))
    
    return tf.nn.relu((conv2d_valid(input, W_r)+b_r)*(conv2d_valid(input, W_g)+b_g)+conv2d_valid(input*input, W_b)+c) 

def Quad_conv_layer_same(input, shape):
    
    W_r = weight_variable_Wr(shape)
    #W_g = weight_variable_Wg(shape)
    W_g = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    #W_b = weight_variable_Wb(shape)
    W_b = tf.Variable(tf.constant(0, shape=shape,dtype=tf.float32))
    b_r = tf.Variable(tf.constant(-0.04, shape=[shape[3]],dtype=tf.float32))
    b_g = tf.Variable(tf.constant(1, shape=[shape[3]],dtype=tf.float32))
    c = tf.Variable(tf.constant(0, shape=[shape[3]],dtype=tf.float32))
    
    return tf.nn.relu((conv2d_same(input, W_r)+b_r)*(conv2d_same(input, W_g)+b_g)+conv2d_same(input*input, W_b)+c) 


          
          


data_train,label_train = read_h5('Mayo_2D_training.h5')
data_test,label_test = read_h5('Mayo_2D_testing.h5')

data_train = np.reshape(data_train,(-1,64,64)) 
label_train = np.reshape(label_train,(-1,64,64))

data_test = np.reshape(data_test,(-1,64,64)) 
label_test = np.reshape(label_test,(-1,64,64))

#%%
with tf.device('/device:GPU:0'):
 x_noised = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
 x_authentic = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
 Study_rate = tf.placeholder(tf.float32)
 
 encode_conv1 = Quad_conv_layer_same(x_noised, shape=[3, 3, 1, 15])
 encode_conv2 = Quad_conv_layer_same(encode_conv1,shape=[3, 3,15,15])
 encode_conv3 = Quad_conv_layer_same(encode_conv2,shape=[3, 3,15,15])

 encode_conv4 = Quad_conv_layer_same(encode_conv3,shape=[3, 3,15,15])

 encode_conv5 = Quad_conv_layer_valid(encode_conv4,shape=[3, 3,15,15])

 decode_conv4 = tf.nn.relu(Quad_deconv_layer_valid_linear(encode_conv5,shape=[3, 3,15,15],outputshape=tf.shape(encode_conv4))+encode_conv4)
 decode_conv3 = Quad_deconv_layer_same(decode_conv4,shape=[3, 3,15,15],outputshape=tf.shape(encode_conv3))

 decode_conv2 = tf.nn.relu(Quad_deconv_layer_same_linear(decode_conv3,shape=[3, 3,15,15],outputshape=tf.shape(encode_conv2))+encode_conv2)
 decode_conv1 = Quad_deconv_layer_same(decode_conv2,shape=[3, 3,15,15],outputshape=tf.shape(encode_conv1))



 x_output = tf.nn.relu(Quad_deconv_layer_same_linear(decode_conv1,shape=[3, 3,1,15],outputshape=tf.shape(x_noised))+x_noised)

 cost = tf.reduce_mean(tf.square(tf.subtract(x_output, x_authentic)))

 optimizer = tf.train.AdamOptimizer(learning_rate=Study_rate).minimize(cost)


Batch_size = 50
Iter_num = 10
Train_Number = 64000
Test_Number = 64000

data_train = data_train[0:64000]
label_train = label_train[0:64000]
data_test = data_test[0:64000]
label_test = label_test[0:64000]

shape = np.int(Train_Number/Batch_size)

loss = np.zeros((1,shape))
validation_loss = np.zeros((1,20))

#batch = x_test[i*Batch_size:(i+1)*Batch_size,:,:]
Evaluate = np.zeros((Test_Number,28,28,1))

data_validation = data_test[0:10000]
data_validation = normalize_y(data_validation,lower = -300.0, upper = 300.0)
label_validation = label_test[0:10000]
label_validation = normalize_y(label_validation,lower = -300.0, upper = 300.0)

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    
    sess.run(tf.global_variables_initializer())
    #x_ou = sess.run(x_output,feed_dict={x:batch})

    NumOfParam = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print(NumOfParam)



    #loss[] = sess.run(cost, feed_dict={x:x_train[i*Batch_size:(i+1)*Batch_size,:]})
    for iteration in np.arange(Iter_num):
      #i=0  

      print(iteration)
      for i in np.arange(shape):
        
        data_train_batch = data_train[i*Batch_size:(i+1)*Batch_size]  
        data_train_batch = normalize_y(data_train_batch,lower = -300.0, upper = 300.0)
        label_train_batch = label_train[i*Batch_size:(i+1)*Batch_size]  
        label_train_batch = normalize_y(label_train_batch,lower = -300.0, upper = 300.0)
        
        
        
        sess.run(optimizer, feed_dict={x_noised:data_train_batch, x_authentic:label_train_batch,Study_rate:0.4e-3})
        #loss[0,i+iteration*shape] = sess.run(cost, feed_dict={x_noised:data_train_batch, x_authentic:label_train_batch,Study_rate:1.5e-3})
        loss[0,i] = sess.run(cost, feed_dict={x_noised:data_train_batch, x_authentic:label_train_batch}) 
     
        if i % 50 == 0:
          #print(loss[0,i+iteration*shape])
          print(loss[0,i])
      
      
      for i in np.arange(10):
        validation_loss[0,iteration] = validation_loss[0,iteration] + sess.run(cost, feed_dict={x_noised:data_validation[i*1000:(i+1)*1000], x_authentic:label_validation[i*1000:(i+1)*1000]})

      print('validation:',validation_loss[0,iteration])


 
    for iteration in np.arange(10):
      #i=0  

      print(iteration)
      for i in np.arange(shape):
        
        data_train_batch = data_train[i*Batch_size:(i+1)*Batch_size]  
        data_train_batch = normalize_y(data_train_batch,lower = -300.0, upper = 300.0)
        label_train_batch = label_train[i*Batch_size:(i+1)*Batch_size]  
        label_train_batch = normalize_y(label_train_batch,lower = -300.0, upper = 300.0)
        
        
        
        sess.run(optimizer, feed_dict={x_noised:data_train_batch, x_authentic:label_train_batch,Study_rate:0.2e-3})
        #loss[0,i+iteration*shape] = sess.run(cost, feed_dict={x_noised:data_train_batch, x_authentic:label_train_batch,Study_rate:1.5e-3})
        loss[0,i] = sess.run(cost, feed_dict={x_noised:data_train_batch, x_authentic:label_train_batch})


        if i % 50 == 0:
          #print(loss[0,i+iteration*shape])
          print(loss[0,i])

      for i in np.arange(10):
        validation_loss[0,iteration] = validation_loss[0,iteration] + sess.run(cost, feed_dict={x_noised:data_validation[i*1000:(i+1)*1000], x_authentic:label_validation[i*1000:(i+1)*1000]})

      print('validation:',validation_loss[0,iteration+10])

    save_path = saver.save(sess, "QuadraticComparison/SmallQuadraticAE/QuadraticAE_small.ckpt")    
    




