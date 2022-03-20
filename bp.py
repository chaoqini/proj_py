#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy
from keras.datasets import mnist
from matplotlib import pyplot as plt 

def pt(x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x

##%matplotlib inline 
#import matplotlib.pyplot as plt
#from tensorflow import keras
#
#def ds_imshow(im_data, im_label):
#    plt.figure(figsize=(5,5))
#    for i in range(len(im_data)):
#        plt.subplot(5,5,i+1)
#        plt.xticks([])
#        plt.yticks([])
#        plt.grid(False)
#        plt.imshow(im_data[i], cmap=plt.cm.binary)
#        plt.xlabel(im_label[i])
#    plt.show()
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
##ds_imshow(x_train[:15].reshape((15,28,28)), y_train[:15]) 
#ds_imshow(x_train[:1].reshape((1,28,28)), y_train[:1]) 

#from keras.utils import to_categorical
#from keras import models, layers, regularizers
#from keras.optimizers import RMSprop
#from keras.datasets import mnist
#import matplotlib.pyplot as plt

# 加载数据集
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
x=mnist.load_data()
#pt(train_images)
#pt(x)
pt(x[0][1][0])
#print(train_images.shape, test_images.shape)
#print(train_images[0])
#print(train_labels[0])
#plt.imshow(train_images[0])
#plt.show()
