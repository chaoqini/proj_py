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
def softmax_d(x): 
#    print('softmax x shape:',x.shape)
    y=softmax(x)
#    print('softmax y shape:',y.shape)
    d_y=[]
    for i in y:
        i=np.squeeze(i)
#        print('i shape:',i.shape)
#        print('i0 shape:',i[0].shape)
        diag=np.diag(i)
        outer=np.outer(i,i)
        d_y.append(diag-outer)
#        print('diag shape:',diag.shape)
#        print('outer shape:',outer.shape)
    d_y=np.array(d_y)
#    print('d_y shape:',d_y.shape)
#    print('d_y.T shape:',d_y.T.shape)
#    print('d_y.transpose(0,2,1) shape:',d_y.transpose(0,2,1).shape)
#    exit(0)
    return d_y

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
#    gl=[log_loss,sqr_loss];gl_d=[log_loss_d,sqr_loss_d]
    gl=[sqr_loss,log_loss];gl_d=[sqr_loss_d,log_loss_d]


    def FP(X,params,isop=0):
        print('FP X shape : ',X.shape)
#        X=X.transpose(0,2,1)
#        print('FP X transpose shape : ',X.shape)
#        print(X.shape)
        b0=params['b0'].reshape(-1,1)
        b1=params['b1'].reshape(-1,1)
        w1=params['w1']
        Z0=X+b0
        A0=ann.g[0](Z0)
        Z1=w1@A0+b1
        A1=softmax(Z1)
        Y=A1
        print('FP b0 shape: ',b0.shape)
        print('FP w1 shape: ',w1.shape)
        print('FP b1 shape: ',b1.shape)
        print('FP X shape: ',X.shape)
        print('FP Z0 shape: ',Z0.shape)
        print('FP A0 shape: ',A0.shape)
        print('FP Z1 shape: ',Z1.shape)
        print('FP A1 shape: ',A1.shape)
        print('FP Y shape: ',Y.shape)
        if isop==0:
            return Y
        else:
#        op={'params':params,'z0':z0,'a0':a0,'z1':z1,'a1':a1}
            OP={'Z0':Z0,'A0':A0,'Z1':Z1,'A1':A1}
            return (Y,OP)


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


    def BP(X,params,LAB):
        print('BP X shape:',X.shape)
#        X=X.transpose(0,2,1)
#        print('BP X.transpose(0,2,1) shape:',X.shape)
        (Y,OP)=ann.FP(X,params,1)
        Z0=OP['Z0']
        A0=OP['A0']
        Z1=OP['Z1']
        A1=OP['A1']
        w1=params['w1']
#        print(LAB.shape)
##        print(LAB[0])
#        print(Y.shape)
#        print(Y[0].shape)
        YL=[]
        for i in range(LAB.shape[0]):
            YL.append(np.eye(Y.shape[1])[LAB[i]].reshape(-1,1))
        YL=np.array(YL)
        d_A1=ann.gl_d[0](Y,YL)
        d_Z1=(softmax_d(Z1).transpose(0,2,1))@d_A1
        d_Z0=ann.g_d[0](Z0)*(w1.T@d_Z1)
        d_w1=d_Z1@A0.transpose(0,2,1)
        d_b1=d_Z1
        d_b0=d_Z0
        d_w1=np.mean(d_w1,axis=0)
        d_b1=np.mean(d_b1,axis=0)
        d_b0=np.mean(d_b0,axis=0)
        grad={'d_w1':d_w1,'d_b1':d_b1,'d_b0':d_b0}

#        print('YL shape:',YL.shape)
#        print('Y shape:',Y.shape)
#        print('d_A1 shape:',d_A1.shape)
#        print('softmax_d(Z1) shape:',softmax_d(Z1).shape)
#        print('d_Z1 shape:',d_Z1.shape)
#        print('d_Z0 shape:',d_Z0.shape)
#        print('A1 shape:',A1.shape)
#        print('d_w1 shape:',d_w1.shape)
#        print('d_b1 shape:',d_b1.shape)
#        print('d_b0 shape:',d_b0.shape)
        return grad


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
        d_z1=(softmax_d(z1).T)@d_a1
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
def batch(params,batch=0,num_batch=0,lr=1,isplot=0,isslope=0):
    if batch==0: batch=100
    max_num_batch=int(mnist.train_num/batch)
    if num_batch<1: num_batch=max_num_batch
    num_batch=min(max_num_batch,int(num_batch))

#    print(mnist.train_img.shape)
#    print(mnist.train_img.shape[0])
#    print(mnist.train_img.shape[1],mnist.train_img.shape[2])
#    num_batch=int(mnist.train_img.shape[0]/batch)
#    nn=mnist.train_img.shape[0]-mnist.train_img.shape[0]%batch
#    print(num_batch)
#    X=np.zeros((nn,mnist.train_img.shape[1],mnist.train_img.shape[2])).reshape(-1,batch,1,28*28)
#    print(X.shape)
#    X[0]=mnist.train_img[0:batch]
#    X=np.empty()
#    X.append(mnist.train_img[0*batch:1*batch-1])
#    X.append(mnist.train_img[1*batch:2*batch-1])
    X=[];LAB=[]
    num_batch=int(min(num_batch,len(mnist.train_img)/batch))
#    print('num_batch=',num_batch)
    for n in range(num_batch):
        X.append(mnist.train_img[n*batch:(n+1)*batch-1])
        LAB.append(mnist.train_lab[n*batch:(n+1)*batch-1])
#        print(X[n].shape)
#        print(LAB[n].shape)
    X=np.array(X)
    LAB=np.array(LAB)
#    print('X shape:',X.shape)
##    print(X[0].shape)
#    print('LAB shape:',LAB.shape)
#    print(LAB[0].shape)
#    ann.FP(X[0],params)
#    if isslope==0 :
#        print('-- use derivative grade function')
#        grad_function=ann.BP
#    else:
#        print('-- use slope grade function')
#        grad_function=ann.k
#    GRAD=grad_function(X[0],params,LAB[0])
#    print(GRAD.keys())

#    for i in 

#    ann.BP(X[0],params,LAB[0])
#    print('len(X)=',len(X))
    for i in range(len(X)):
        print('iteration number=:',i)
        grad=ann.BP(X[i],params,LAB[i])
        params=ann.update_params(params,grad,lr)
    return params


#   X=mnist.train_img[:] 

#    loss=[]
#    for i in range(num_batch):
#        nb=batch*i
#        x=mnist.train_img[nb]
#        lab=mnist.train_lab[nb]
#        grad_acc=grad_function(x,params,lab)
##        print('='*10,' %s/%s :'%(i+1,num_batch))
#        loss_acc=0
#        for j in range(1,batch):
#            if isslope!=0 :
#                print('train number is  %s/%s at %s/%s batchs'%(j+1,batch,i+1,num_batch))
#            n=nb+j
#            x=mnist.train_img[n]
#            lab=mnist.train_lab[n]
#            grad=grad_function(x,params,lab)
#            for k in grad.keys():
#                grad_acc[k]+=grad[k]
#            loss_acc+=ann.gl[0](x,params,lab)
#        for k in grad_acc.keys():
#            grad_acc[k]=grad_acc[k]/batch
#        params=ann.update_params(params,grad_acc,lr)
#        loss_acc=loss_acc/batch
#        loss.append(loss_acc)
#    if isplot!=0:
#        plt.plot(loss)
#        plt.ylabel('Loss')
#        plt.xlabel('Iterations /%s'%batch)
#        var_title=(ann.g[0].__name__,ann.gl[0].__name__,lr)
#        title='Active = %s\n Loss function = %s\n Learning rate = %s\n'%var_title
#        plt.title(title)
#        plt.show()
#    return params
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
params=batch(params,2,2,.01,1)
(valid_per,loss_avg)=valid(params)
#show(num)
#with open('p3.pkl', 'wb') as f: pickle.dump(params,f)
## ==========

## ==========
#num=np.random.randint(mnist.test_num)
#x=mnist.train_img[num]
#lab=mnist.train_lab[num]
#ann.cmp(x,params,lab,'d_w1',1e-15)
##ann.cmp(x,params,lab,'d_w1')
#show(num)
## ==========
