#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
from matplotlib import pyplot as plt 
import pickle
import sys


## ==========
## ==========
def tanh(x): return np.tanh(x)
def tanh_d(x):
    y=tanh(x)
    return 1-y**2
def relu(x): return np.maximum(0,x)
def relu_d(x): return (np.sign(x)+1)/2
def softmax(x): 
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def d_softmax(x): 
    y=softmax(x)
    return np.diag(y[:,0])-np.outer(y,y)

def sqr_loss(x,params,lab):
    y=ann.fp(x,params)
    yl=np.eye(y.shape[0])[lab].reshape(-1,1)
    l=(y-yl).T@(y-yl)
    l=np.sum((y-yl)*(y-yl))
    return l
def sqr_loss_d(y,yl):
    return 2*(y-yl)
def log_loss(x,params,lab,a=1e-23):
    y=ann.fp(x,params)
    yl=np.eye(y.shape[0])[lab].reshape(-1,1)
#    l=-(yl*np.log(y+a)+(1-yl)*np.log(1-y+a))
    l=-np.sum(yl*np.log(y+a)+(1-yl)*np.log(1-y+a))
    return l
def log_loss_d(y,yl,a=1e-23):
    dl=-(yl/(y+a)+(1-yl)/(1-y+a))
    return dl
## ==========
## ==========


## ==========
## ==========
class ann:
    g=[tanh,relu];g_d=[tanh_d,relu_d]
#    g=[relu,tanh];g_d[relu_d,tanh_d]
    gl=[log_loss,sqr_loss];gl_d=[log_loss_d,sqr_loss_d]
#    gl=[sqr_loss,log_loss];gl_d=[sqr_loss_d,log_loss_d]
    def fp(x,params,isop=0):
        x=x.reshape(-1,1)
        b0=params['b0'].reshape(-1,1)
        b1=params['b1'].reshape(-1,1)
        w1=params['w1']
        z0=x+b0
        a0=ann.g[0](z0)
        z1=w1@a0+b1
        a1=softmax(z1)
        y=a1
        if isop==0:
            return y
        else:
#        op={'params':params,'z0':z0,'a0':a0,'z1':z1,'a1':a1}
            op={'z0':z0,'a0':a0,'z1':z1,'a1':a1}
            return (y,op)
## ==========
    def bp(x,params,lab):
        (y,op)=ann.fp(x,params,1)
        z0=op['z0'].reshape(-1,1)
        a0=op['a0'].reshape(-1,1)
        z1=op['z1'].reshape(-1,1)
        a1=op['a1'].reshape(-1,1)
        w1=params['w1']
        yl=np.eye(y.shape[0])[lab].reshape(-1,1)
        d_a1=ann.gl_d[0](y,yl)
        d_z1=(d_softmax(z1).T)@d_a1
        d_z0=ann.g_d[0](z0)*(w1.T@d_z1)
        d_w1=d_z1@a0.T
        d_b1=d_z1
        d_b0=d_z0
        grad={'d_w1':d_w1,'d_b1':d_b1,'d_b0':d_b0}
        return grad
## ==========
    def k(x,params,lab,h=1e-6):
        slope={}
        pp = copy.deepcopy(params) 
        for (k,v) in pp.items():
            k='d_'+k
            slope[k]=np.zeros(v.shape)
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    v[i,j]-=h
                    l1=ann.gl[0](x,pp,lab)
                    v[i,j]+=2*h
                    l2=ann.gl[0](x,pp,lab)
                    v[i,j]-=h
                    slp=(l2-l1)/(2*h)
                    slope[k][i,j]=slp
        return slope
    def cmp(x,parms,lab,k='d_b1',h=1e-6):
        param_d=ann.bp(x,params,lab)
        param_k=ann.k(x,params,lab)
        ppd = param_d[k]
        ppk = param_k[k]
        (rd,cd) = ppd.shape
        (rk,ck) = ppk.shape
        (prd,pcd)=(int(rd/2),int(cd/2))
        (prk,pck)=(int(rk/2),int(ck/2))
        print('%s derivative:'%k)
        print(ppd[prd-5:prd+5,pcd-5:pcd+5])
        print('%s slope:'%k)
        print(ppk[prk-5:prk+5,pck-5:pck+5])

## ==========
#    def cost(x,params,lab):
#        y=ann.fp(x,params).reshape(-1,1)
#        yl=np.eye(y.shape[0])[lab].reshape(-1,1)
#        loss = -yl*np.log(y+ann.log_dv)-(1-yl)*np.log(1-y+ann.log_dv)
##        m=y.shape[1]
#        m=1
#        cost=np.sum(loss)/m
#        return cost
## ==========
    def update_params(params,grad,lr=1):
        b0=params['b0'].reshape(-1,1)
        b1=params['b1'].reshape(-1,1)
        w1=params['w1'].reshape(b1.shape[0],-1)
        d_b0=grad['d_b0']
        d_b1=grad['d_b1']
        d_w1=grad['d_w1']
        b0-=lr*d_b0
        b1-=lr*d_b1
        w1-=lr*d_w1
        params={'b0':b0,'b1':b1,'w1':w1}
        return params
## ==========
## ==========



## ==========
#mnist.train_num=50000
def batch(params,batch=50,num_batch=0,lr=1,isplot=0,isslope=0):
    if batch==0: batch=100
    max_num_batch=int(mnist.train_num/batch)
    if num_batch<1: num_batch=max_num_batch
    num_batch=min(max_num_batch,int(num_batch))
    if isslope==0 :
        print('-- use derivative grade function')
        grad_function=ann.bp
    else:
        print('-- use slope grade function')
        grad_function=ann.k
    loss=[]
    for i in range(num_batch):
        nb=batch*i
        x=mnist.train_img[nb]
        lab=mnist.train_lab[nb]
        grad_acc=grad_function(x,params,lab)
#        print('='*10,' %s/%s :'%(i+1,num_batch))
        loss_acc=0
        for j in range(1,batch):
            if isslope!=0 :
                print('train number is  %s/%s at %s/%s batchs'%(j+1,batch,i+1,num_batch))
            n=nb+j
            x=mnist.train_img[n]
            lab=mnist.train_lab[n]
            grad=grad_function(x,params,lab)
            for k in grad.keys():
                grad_acc[k]+=grad[k]
            loss_acc+=ann.gl[0](x,params,lab)
        for k in grad_acc.keys():
            grad_acc[k]=grad_acc[k]/batch
        params=ann.update_params(params,grad_acc,lr)
        loss_acc=loss_acc/batch
        loss.append(loss_acc)
    if isplot!=0:
        plt.plot(loss)
        plt.ylabel('Loss')
        plt.xlabel('Iterations /%s'%batch)
        var_title=(ann.g[0].__name__,ann.gl[0].__name__,lr)
        title='Active = %s\n Loss function = %s\n Learning rate = %s\n'%var_title
        plt.title(title)
        plt.show()
    return params
## ==========
def valid(params):
    correct=[]
    loss_acc=0
    for i in range(mnist.valid_img.shape[0]):
        x=mnist.valid_img[i]
        lab=mnist.valid_lab[i]
        loss=ann.gl[0](x,params,lab)
        loss_acc+=loss
        is1=(ann.fp(x,params).argmax()==lab)
        correct.append(is1)
    valid_per=correct.count(1)/len(correct)
    loss_avg=loss_acc/(mnist.valid_img.shape[0])
    print('valid percent is : %.2f%%'%(valid_per*100))
    print('average loss is : %s'%(loss_avg))
    return (valid_per,loss_avg)
## ==========
def show(n=0):
    x=mnist.test_img[n]
    lab=mnist.test_lab[n]
    predict=np.argmax(ann.fp(x,params))
    print('Real lab number is :\t%s'%lab)
    print('Precdict number is :\t%s'%predict)
    img=x.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()
## ==========
## ==========



## ==========
#with open('p2.pkl', 'wb') as f: pickle.dump(params,f)
with open('p3.pkl', 'rb') as f: params_saved=pickle.load(f)
nx=28*28; ny=10
params_init={'b0':0*np.ones((nx,1)),'b1':0*np.ones((ny,1)),'w1':1*np.ones((ny,nx))}
#params=params_saved
params=params_init
## ==========

## ==========
##  training
## ==========
#params=batch(params,10,2,1)
#params=batch(params,100,300)
#params=batch(params,200)
#params=batch(params,200,0,.01,1)
params=batch(params,10,0,.001,1)
(valid_per,loss_avg)=valid(params)
#show(num)
#with open('p3.pkl', 'wb') as f: pickle.dump(params,f)
## ==========

## ==========
num=np.random.randint(mnist.test_num)
x=mnist.train_img[num]
lab=mnist.train_lab[num]
ann.cmp(x,params,lab,'d_w1',1e-15)
##ann.cmp(x,params,lab,'d_w1')
#show(num)
## ==========
