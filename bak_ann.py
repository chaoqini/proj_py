#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
from matplotlib import pyplot as plt 
import pickle
import sys
import time


## ==========
## ==========
def tanh(x): return np.tanh(x)
def tanh_d(x):
    y=tanh(x)
    return 1-y**2
#def relu(x): return np.maximum(0,x)
def relu(x,kn=1e-3):
    y=copy.deepcopy(x).reshape(-1)
    yt=x.reshape(-1)
    y[yt<0]=y[yt<0]*kn
    y[yt>=0]=y[yt>=0]
    y=y.reshape(x.shape)
    return y
def relu_d(x,kn=1e-3):
    y=copy.deepcopy(x).reshape(-1)
    yt=x.reshape(-1)
    y[yt<0]=kn
    y[yt>=0]=1
    y=y.reshape(x.shape)
    return y
def softmax(X): 
    if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
    assert(X.ndim==3)
    exp=np.exp(X-np.max(X,axis=1,keepdims=1))
    expsum=np.sum(exp,axis=1,keepdims=1)
    return exp/expsum
#def softmax_d(X): 
#    OUT=[]
#    for i in range(len(X)):
#        y=softmax(X[i])    
#        diag=np.diag(np.squeeze(y))
#        outer=np.outer(y,y)
#        OUT.append(diag-outer)
#    OUT=np.array(OUT)
#    return OUT
#def sqr_loss(x,params,lab):
#    y=ann.fp(x,params)
#    yl=np.eye(y.shape[0])[lab].reshape(-1,1)
#    l=(y-yl).T@(y-yl)
#    l=np.sum((y-yl)*(y-yl))
#    return l
#def sqr_loss_d(Y,YL):
#    OUT=[]
#    for i in range(len(YL)):
#        OUT.append(2*(Y[i]-YL[i]))
#    OUT=np.array(OUT)
#    return  OUT
def log_loss(X,params,LAB,isvalid=0):
    Y=ann.fp(X,params)
    meye=np.array([np.eye(Y.shape[1])]*len(LAB))
    lab=LAB.reshape(-1)
    nbatch=np.arange(len(meye))
    YL=meye[nbatch,lab,:]
    YL=YL.reshape(YL.shape+tuple([1]))
#    print('log_loss: Y=',Y)
    LOSS=-YL*np.log(Y)
    cost=np.sum(LOSS)/len(LOSS)
    if isvalid==0:
        return cost
    else:
        y1d_max=np.max(Y,axis=1,keepdims=1)
        Y=np.trunc(Y/y1d_max)
        cmp=Y==YL
        correct=np.trunc(np.sum(cmp,axis=1)/cmp.shape[1])
        valid_per=correct.sum()/len(correct)
        return (cost,valid_per,correct)


## ==========
## ==========
class ann:
#    g=(tanh,relu,softmax,log_loss);g_d=(tanh_d,relu_d,softmax_d,log_loss_d)
#    g=(tanh,tanh,softmax,log_loss);g_d=(tanh_d,tanh_d,None,None)
    g=(tanh,relu,softmax,log_loss);g_d=(tanh_d,relu_d,None,None)

    def fp(X,params,isop=0):
        if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
        assert(X.ndim==3)
        b0=params['b0'].reshape(-1,1)
        b1=params['b1'].reshape(-1,1)
#        b2=params['b2'].reshape(-1,1)
        w1=params['w1']
#        w2=params['w2']
        Z0=X+b0
        A0=ann.g[0](Z0)
        Z1=w1@A0+b1
        A1=ann.g[-2](Z1)
#        Z2=w2@A1+b2
#        A2=ann.g[2](Z2)
        Y=A1
#        print('fp: Y=',Y)
        if isop==0:
            return Y
        else:
#            OP={'Z':[Z0,Z1,Z2],'A':[A0,A1,A2]}
            OP={'Z':[Z0,Z1],'A':[A0,A1]}
            return (Y,OP)

## ==========
    def bp(X,params,LAB):
        if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
        if LAB.ndim==2: LAB=LAB.reshape(tuple([1])+LAB.shape)
        assert(X.ndim==3)
        assert(LAB.ndim==3)
        (Y,OP)=ann.fp(X,params,1)
        assert(Y.ndim==3)
        Z0=OP['Z'][0]
        Z1=OP['Z'][1]
#        Z2=OP['Z'][2]
        A0=OP['A'][0]
        A1=OP['A'][1]
#        A2=OP['A'][2]
        w1=params['w1']
#        w2=params['w2']
        meye=np.array([np.eye(Y.shape[1])]*len(LAB))
        assert(meye.ndim==3)
        lab=LAB.reshape(-1)
        assert(lab.ndim==1)
        nbatch=np.arange(len(meye))
        YL=meye[nbatch,lab,:]
        YL=YL.reshape(YL.shape+(1,))
        d_Z1=Y-YL
#        d_Z1=ann.g_d[1](Z1)*(w2.T@d_Z2)
        d_Z0=ann.g_d[0](Z0)*(w1.T@d_Z1)
#        d_W2=d_Z2@A1.transpose(0,2,1)
        d_W1=d_Z1@A0.transpose(0,2,1)
#        d_B2=d_Z2
        d_B1=d_Z1
        d_B0=d_Z0
#        d_w2=np.mean(d_W2,axis=0)
        d_w1=np.mean(d_W1,axis=0)
#        d_b2=np.mean(d_B2,axis=0)
        d_b1=np.mean(d_B1,axis=0)
        d_b0=np.mean(d_B0,axis=0)
#        grad={'d_b0':d_b0,'d_b1':d_b1,'d_b2':d_b2,'d_w1':d_w1,'d_w2':d_w2}
        grad={'d_b0':d_b0,'d_b1':d_b1,'d_w1':d_w1}
        return grad

## ==========
    def slope(x,params,lab,dv=1e-5):
#        print('slope:')
        slp={}
        pt=copy.deepcopy(params) 
        for (k,v) in pt.items():
#            print('slope: k=',k)
#            print('slope: v=',v)
            d_k='d_'+k
#            print('slope: k_d=',k_d)
            slp[d_k]=np.zeros(v.shape)
            for i in range(len(v)):
                for j in range(len(v[i])):
                    vb=v[i,j]
                    v[i,j]=vb-dv
                    l1=ann.g[-1](x,pt,lab)
                    v[i,j]=vb+dv
                    l2=ann.g[-1](x,pt,lab)
                    v[i,j]=vb
                    kk=(l2-l1)/(2*dv)
                    slp[d_k][i,j]=kk
        iseq=1
        for k in params.keys():
#            iseq=iseq&(np.any(pt[k]==params[k])) 
            iseq=iseq&(np.all(pt[k]==params[k])) 
        assert(iseq==1)
        return slp

    def grad_check(x,params,lab,dv=1e-5):
#        print('grad_check:')
        y1=ann.bp(x,params,lab)
        y2=ann.slope(x,params,lab,dv)
        abs_error={};ratio_error={}
        for (k,v) in y1.items():
            v1=v
            v2=y2[k]
            l2_v1=np.linalg.norm(v1)
            l2_v2=np.linalg.norm(v2)
            l2_v1d2=np.linalg.norm(v1-v2)
            abs_error[k]=l2_v1d2
            ratio_error[k]=l2_v1d2/(l2_v1+l2_v2)
#            print('grad_check: %s abs_err= %s'%(k,abs_error[k]))
#            print('grad_check: %s ratio_err= %s'%(k,ratio_error[k]))
        print('grad_check: abs_error=',abs_error)
        print('grad_check: ratio_error=',ratio_error)
        return (ratio_error,abs_error)
## ==========
    def update_params(params,grad,lr=0.1):
        for k in params.keys():
            params[k]-=lr*grad['d_'+k]
        return params

## ==========
#mnist.train_num=50000
def batch(params,batch=0,batches=0,lr=0,isplot=0,istime=0):
    for k in params.keys():
        print( 'params %s.shape='%k,params[k].shape)
#    for i in ann.g:
    for i in range(len(ann.g)):
        print('active function g[%d] is:'%i,ann.g[i].__name__)
    if batch<1: batch=100
    max_batches=int(len(mnist.train_img)/batch)
    if batches<1: batches=max_batches
    batches=min(max_batches,int(batches))
    if lr==0: lr=0.1
    X=mnist.train_img[:batch*batches]
    LAB=mnist.train_lab[:batch*batches]
    X=X.reshape((-1,batch)+X.shape[1:3])
    LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
    cost=[]
    print('Batch training running ...')
    for i in range(len(X)):
        pn=i%(max(int(len(X)/10),1))
        if pn==0 or i==len(X)-1:
            print('iteration number = %s/%s'%(i,len(X)))
            if istime!=0:tb=time.time()
        grad=ann.bp(X[i],params,LAB[i])
        params=ann.update_params(params,grad,lr)
        (cost_i,valid_per,correct)=ann.g[-1](X[i],params,LAB[i],1)
        cost.append(cost_i)
        if (pn==0 or i==len(X)-1) and istime!=0:
            te=time.time()
            tspd=(te-tb)*1000
            print('the spending time of %s/%s batch is %s mS'%(i,len(X),tspd))
    cost=np.array(cost)
    if isplot!=0:
        plt.plot(cost)
        plt.ylabel('Cost')
        plt.xlabel('Iterations /%s'%i)
        var_title=(ann.g[0].__name__,ann.g[-2].__name__,ann.g[-1].__name__,lr)
        title='Active g[0]= %s\n Active g[-2]= %s\n Loss function g[-1]= %s\n \
        Learning rate = %s\n'%var_title
        plt.title(title)
        plt.show()
    return params
## ==========
## ==========
def valid(params,n=0):
    if n==0 : n=mnist.valid_img.shape[0]
    IMG=mnist.valid_img[:n]
    LAB=mnist.valid_lab[:n]
    (cost,valid_per,correct)=ann.g[-1](IMG,params,LAB,1)
    print('L2_b0=',np.linalg.norm(params['b0'])/params['b0'].size)
    print('L2_b1=',np.linalg.norm(params['b1'])/params['b1'].size)
    print('L2_w1=',np.linalg.norm(params['w1'])/params['w1'].size)
    print('valid percent is : %.2f%%'%(valid_per*100))
    return (valid_per,correct) 
## ==========
def valid_train(params,n=0):
    if n==0 : n=mnist.train_img.shape[0]
    IMG=mnist.train_img[:n]
    LAB=mnist.train_lab[:n]
    (cost,valid_per,correct)=ann.g[-1](IMG,params,LAB,1)
    print('L2_b0=',np.linalg.norm(params['b0'])/params['b0'].size)
    print('L2_b1=',np.linalg.norm(params['b1'])/params['b1'].size)
    print('L2_w1=',np.linalg.norm(params['w1'])/params['w1'].size)
    print('train_valid percent is : %.2f%%'%(valid_per*100))
    return (valid_per,correct) 
## ==========

## ==========
def show(n=-1):
    if n==-1: n=np.random.randint(mnist.test_num)
    x=mnist.test_img[n]
    lab=mnist.test_lab[n].squeeze()
    y=ann.fp(x,params)
    y=np.argmax(y)
    print('Real lab number is :\t%s'%lab)
    print('Precdict number is :\t%s'%y)
    img=x.reshape(28,28)
#    plt.imshow(img)
    plt.imshow(img,cmap='gray')
    plt.show()
## ==========
## ==========



## ==========
#with open('ann_p1.pkl', 'rb') as f: params_saved=pickle.load(f)
nx=28*28;ny1=10;ny2=10
#params_init={'b0':0*np.ones((nx,1)),'b1':0*np.ones((ny,1)),'w1':0.01*np.ones((ny,nx))}
b0=0*np.ones((nx,1))
b1=0*np.ones((ny1,1))
b2=0*np.ones((ny2,1))
w1=0.001*np.ones((ny1,nx))
w2=0.001*np.ones((ny2,ny1))
params_init={'b0':b0,'b1':b1,'w1':w1}
#params_init={'b0':b0,'b1':b1,'b2':b2,'w1':w1,'w2':w2}
#params_init={'b0':1e-3*np.ones((nx,1)),'b1':1e-3*np.ones((ny,1)),'w1':0.01*np.ones((ny,nx))}
#print('params_init[b0]=',params_init['b0'])
#params=params_saved
params=params_init
## ==========

## ==========
##  training
print('Training running ...')
## ==========
#params=batch(params,0,0,0,1)
#params=batch(params,100,300)
#params=batch(params,200)
#params=batch(params,200,0,.01,1)
#params=batch(params,1,0,.1,1)
#params=batch(params,30,200,.01,1)
#params=batch(params,10,0,.01,1)
#params=batch(params,20,0,0,1)
#params=batch(params,0,0,0,1,1)
params=batch(params,0,0,0,1,1)
#params=batch(params)
#params=batch(params,20,40,.1,1)
#params=batch(params,3,2,.01,1)
## ==========
#(valid_per,loss_avg)=valid(params,3)
#(valid_per,corrent)=valid(params,12)
## ==========
## valid
print('Valid running ...')
## ==========
(valid_per,correct)=valid(params)
(valid_per2,correct2)=valid_train(params)
## ==========
## ==========
## grade check
print('Grade check running ...')
## ==========
num=np.random.randint(mnist.test_num)
num=11
x=mnist.test_img[num]
lab=mnist.test_lab[num]
#y=ann.fp(x,params)
#y=np.argmax(y)
#k1=ann.slope(x,params,lab)
ann.grad_check(x,params,lab,1e-6)
#ann.grad_check(x,params,lab)
#print('k.keys()=',k1.keys())
#show()
#with open('ann_p1.pkl', 'wb') as f: pickle.dump(params,f)
## ==========

