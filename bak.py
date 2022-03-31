#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
#import json
import pandas as pd
#from matplotlib import pyplot as plt 

def pt(x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x
def convolution(a,b):
#    return  np.trace(a.dot(b.T))
    return (a*b).sum()

def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def d_tanh(x):
    y=tanh(x)
#    return np.diag(1/(np.cosh(x))**2)
#    return np.diag(1-y**2)
#    return 1/(np.cosh(x))**2
    return 1-y**2
def d_softmax(x):
    y=softmax(x)
    return np.diag(y)-np.outer(y,y)
def plt_img(img):
    img=img.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()

## ==========
## ==========
def predict(img,params):
    w=params['w']     # m*n
    b1=params['b1'] # m*1
    b0=params['b0'] # n*1
    yi=tanh(img+b0)  # n*1
    yl=w.dot(yi)+b1 # m*1
    ys=softmax(yl) # m*1
    return ys
def loss(img,lab,params):
    ys=predict(img,params) # m*1
    yr=np.eye(ny)[lab] # m*1
    l=np.dot(ys-yr,ys-yr) # 1*1
    return l
def grad_params(img,lab,params):
    w=params['w']     # m*n
    b1=params['b1'] # m*1
    b0=params['b0'] # n*1
    x=img
    yi=tanh(x+b0)  # n*1
    yl=w.dot(yi)+b1 # m*1
    ys=softmax(yl) # m*1
    yr=np.eye(ny)[lab] # m*1
    l=np.dot(ys-yr,ys-yr) # 1*1
    d_l_ys = 2*(ys-yr)  ## 1*m
    d_ys_yl = d_softmax(yl) ## m*m
    d_yl_yi = w   ## m*n      
    d_yi_b0 = 1-yi**2 #  --m*n
    d_l_yl = d_l_ys.dot(d_ys_yl) # 1*m
    d_l_w = np.outer(d_l_yl,yi) # m*n
    d_l_b1=d_l_yl # 1*m
#    d_l_b0=d_l_yl.dot(d_yl_yi).dot(d_yi_b0) # 
    d_l_b0=d_l_yl.dot(d_yl_yi)*d_yi_b0 # 
    grad_w = -d_l_w
    grad_b1 = -d_l_b1
    grad_b0 = -d_l_b0
    return {'w':grad_w,'b1':grad_b1,'b0':grad_b0}
    ## ==========
    ## dl=a.dw.b => dl/dw = (a.T)*(b.T)_m*n
    ## ==========
    ## dl = dl/dys.dys/dyl.dyl = (dl/dys.dys/dyl).dw.yi = dl/dyl.dw.yi
    ## dl/dw = [(dl/dyl).T]*(yi.T)_m*n

def valid_loss(params):
    loss_accu=0
    for i in range(mnist.valid_num):
        loss_accu+=loss(mnist.valid_img[i],mnist.valid_lab[i],params)
    return loss_accu
def valid_accuracy(params):
    correct=[]
    for i in range(mnist.valid_num):
        correct.append(predict(mnist.valid_img[i],params).argmax()==mnist.valid_lab[i])
    return correct.count(1)/len(correct)
batch_size=100
def train_batch(num_batch,params):
    grad_accu=grad_params(mnist.train_img[batch_size*num_batch+0],mnist.train_lab[batch_size*num_batch+0],params)
    for i in range(1,batch_size):
        temp=grad_params(mnist.train_img[batch_size*num_batch+i],mnist.train_lab[batch_size*num_batch+i],params)
        for k in grad_accu.keys():
            grad_accu[k]+=temp[k]
    for k in grad_accu.keys():
        grad_accu[k] = grad_accu[k]/batch_size
    return grad_accu
def combine(params,grad,lr=1):
    params_temp = copy.deepcopy(params) 
    for k in params_temp.keys():
        params_temp[k] = params_temp[k] + grad[k]*lr
    return params_temp

nx=28*28
ny=10
params_init={'b0':0*np.ones(nx),'b1':0*np.ones(ny),'w':1*np.ones([ny,nx])}
#params_init={'b0':np.random.rand(nx),'b1':np.random.rand(ny),'w':np.random.rand(ny,nx)}
params=params_init
#num_img=np.random.randint(mnist.train_num)
#img=mnist.train_img[num_img]
#lab=mnist.train_lab[num_img]
#pred=predict(img,params)
#l=loss(img,lab,params)
#print("The Image is:\n %s"%np.argmax(pred))
#plt_img(img)

num_batch=int(mnist.train_num/batch_size)
for i in range(num_batch):
#for i in range(500):
    print('running batch : %s/%s'%(i+1,num_batch))
    avggrad_batch=train_batch(i,params)
    params=combine(params,avggrad_batch)
x=valid_accuracy(params)
print(x)
#np.save('p1.npy',params)
#np.savetxt('p1.txt',params)
#np.savetxt('p1.txt',params.values())
#np.savetxt('p1.txt',params['b1'])
#npy_p1=np.load('p1.npy',allow_pickle=1)












