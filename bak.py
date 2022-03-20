#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy
from matplotlib import pyplot as plt 

def pt(self,x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x

matplotlib inline 
import matplotlib.pyplot as plt
from tensorflow import keras

def ds_imshow(im_data, im_label):
    plt.figure(figsize=(10,10))
    for i in range(len(im_data)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(im_data[i], cmap=plt.cm.binary)
        plt.xlabel(im_label[i])
    plt.show()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#ds_imshow(x_train[:15].reshape((15,28,28)), y_train[:15]) 
ds_imshow(x_train[:1].reshape((1,28,28)), y_train[:1]) 
