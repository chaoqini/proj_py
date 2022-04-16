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
def relu_d(x):
    y=copy.deepcopy(x) 
    y[y<0]=0
    y[y>0]=1
    return y
def sigmod(x): return 1/(1+np.exp(-x))
def sigmod_d(x):
    y=sigmod(x)
    return y-y*y

def softmax(X): 
    if X.ndim==2: X=X.reshape((1)+X.shape)
    assert(X.ndim==3)
    exp=np.exp(X-np.max(X,axis=1,keepdims=1))
    expsum=np.sum(exp,axis=1,keepdims=1)
    return exp/expsum

def softmax_d(X): 
    OUT=[]
    for i in range(len(X)):
        y=softmax(X[i])    
        diag=np.diag(np.squeeze(y))
        outer=np.outer(y,y)
        OUT.append(diag-outer)
    OUT=np.array(OUT)
    return OUT

#def sqr_loss(x,params,lab):
#    y=ann.fp(x,params)
#    yl=np.eye(y.shape[0])[lab].reshape(-1,1)
#    l=(y-yl).T@(y-yl)
#    l=np.sum((y-yl)*(y-yl))
#    return l
def sqr_loss_d(Y,YL):
    OUT=[]
    for i in range(len(YL)):
        OUT.append(2*(Y[i]-YL[i]))
    OUT=np.array(OUT)
    return  OUT
def log_loss(X,params,LAB,isvalid=0):
    Y=ann.fp(X,params)
    meye=np.array([np.eye(Y.shape[1])]*len(LAB))
    lab=LAB.reshape(-1)
    nbatch=np.arange(len(meye))
    YL=meye[nbatch,lab,:]
    YL=YL.reshape(YL.shape+tuple([1]))
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
def log_loss_d(Y,YL):
    OUT=[]
    for i in range(len(Y)):
        OUT.append(-YL[i]/Y)
    OUT=np.array(OUT)
    return  OUT
## ==========
## ==========


## ==========
## ==========
class ann:
#    g=(tanh,softmax,sqr_loss);g_d=(tanh_d,softmax_d,sqr_loss_d)
    g=(tanh,softmax,log_loss);g_d=(tanh_d,softmax_d,log_loss_d)
#    g=(relu,softmax,log_loss);g_d=(relu_d,softmax_d,log_loss_d)

    def fp(X,params,isop=0):
        if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
        assert(X.ndim==3)
        b0=params['b0'].reshape(-1,1)
        b1=params['b1'].reshape(-1,1)
        w1=params['w1']
        Z0=X+b0
        A0=ann.g[0](Z0)
        Z1=w1@A0+b1
        A1=ann.g[-2](Z1)
        Y=A1
        if isop==0:
            return Y
        else:
            OP={'Z0':Z0,'A0':A0,'Z1':Z1,'A1':A1}
            return (Y,OP)

## ==========
    def bp(X,params,LAB):
        if X.ndim==2: X=X.reshape((1)+X.shape)
        if LAB.ndim==2: LAB=LAB.reshape((1)+LAB.shape)
        assert(X.ndim==3)
        assert(LAB.ndim==3)
        (Y,OP)=ann.fp(X,params,1)
        assert(Y.ndim==3)
        Z0=OP['Z0']
        A0=OP['A0']
        Z1=OP['Z1']
        A1=OP['A1']
        w1=params['w1']
        meye=np.array([np.eye(Y.shape[1])]*len(LAB))
        assert(meye.ndim==3)
        lab=LAB.reshape(-1)
        assert(lab.ndim==1)
        nbatch=np.arange(len(meye))
        YL=meye[nbatch,lab,:]
        YL=YL.reshape(YL.shape+(1,))
        d_Z1=Y-YL
        d_Z0=ann.g_d[0](Z0)*(w1.T@d_Z1)
        d_W1=d_Z1@A0.transpose(0,2,1)
        d_B1=d_Z1
        d_B0=d_Z0
        d_w1=np.mean(d_W1,axis=0)
        d_b1=np.mean(d_B1,axis=0)
        d_b0=np.mean(d_B0,axis=0)
        grad={'d_w1':d_w1,'d_b1':d_b1,'d_b0':d_b0}
        return grad

## ==========
    def slope(x,params,lab,h=1e-6):
        print('slope:')
        slp={}
        pt = copy.deepcopy(params) 
#        for (k,v) in pt.items():
#            k='d_'+k
#            slp[k]=np.zeros(v.shape)
#            for i in range(len(v)):
#                for j in range(len(v[i])):
#                    v[i,j]-=h
#                    l1=ann.g[-1](x,pt,lab)
#                    v[i,j]+=2*h
#                    l2=ann.g[-1](x,pt,lab)
#                    v[i,j]-=h
#                    kk=(l2-l1)/(2*h)
#                    slp[k][i,j]=kk
#        assert(pt.values()==params.values())
        print(pt.values()==params.values())
#        print('pt[b1]=',pt['b1'])
#        print('params[b1]=',params['b1'])
#        print('pt[b0]=',pt['b0'])
#        print('params[b0]=',params['b0'])
#        print('pt[w1]=',pt['w1'])
#        print('params[w1]=',params['w1'])
#        tt=pt.keys()-params.keys()
#        tt=cmp(pt,params)
#        tt=pt.all()==params.all()
#        print(tt)
#        print(type(tt))
        return slp

#    def k(x,params,lab,h=1e-6):
#        slope={}
#        pp = copy.deepcopy(params) 
#        for (k,v) in pp.items():
#            k='d_'+k
#            slope[k]=np.zeros(v.shape)
#            for i in range(v.shape[0]):
#                for j in range(v.shape[1]):
#                    v[i,j]-=h
#                    l1=ann.g[-1](x,pp,lab)
#                    v[i,j]+=2*h
#                    l2=ann.g[-1](x,pp,lab)
#                    v[i,j]-=h
#                    slp=(l2-l1)/(2*h)
#                    slope[k][i,j]=slp
#        return slope
#    def cmp(x,parms,lab,k='d_b1',h=1e-6):
#        param_d=ann.bp(x,params,lab)
#        param_k=ann.k(x,params,lab)
#        ppd = param_d[k]
#        ppk = param_k[k]
#        (rd,cd) = ppd.shape
#        (rk,ck) = ppk.shape
#        (prd,pcd)=(int(rd/2),int(cd/2))
#        (prk,pck)=(int(rk/2),int(ck/2))
#        print('%s derivative:'%k)
#        print(ppd[prd-5:prd+5,pcd-5:pcd+5])
#        print('%s slope:'%k)
#        print(ppk[prk-5:prk+5,pck-5:pck+5])
#
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
def batch(params,batch=0,batches=0,lr=0,isplot=0):
    if batch<1: batch=100
    max_batches=int(len(mnist.train_img)/batch)
    if batches<1: batches=max_batches
    batches=min(max_batches,int(batches))
    if lr==0: lr=1
    X=mnist.train_img[:batch*batches]
    LAB=mnist.train_lab[:batch*batches]
    X=X.reshape((-1,batch)+X.shape[1:3])
    LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
    cost=[]
    for i in range(len(X)):
        if i%(int(len(X)/10))==0 or i==len(X)-1:
            print('iteration number=:%s/%s'%(i,len(X)))
        grad=ann.bp(X[i],params,LAB[i])
        params=ann.update_params(params,grad,lr)
        (cost_i,valid_per,correct)=log_loss(X[i],params,LAB[i],1)
        cost.append(cost_i)
    cost=np.array(cost)
    if isplot!=0:
        plt.plot(cost)
        plt.ylabel('Cost')
        plt.xlabel('Iterations /%s'%i)
        var_title=(ann.g[0].__name__,ann.g[-1].__name__,lr)
        title='Active g[0]= %s\n Loss function g[-1]= %s\n Learning rate = %s\n'%var_title
        plt.title(title)
        plt.show()
    return params

## ==========
def valid(params,n=0):
    if n==0 : n=mnist.valid_img.shape[0]
    IMG=mnist.valid_img[:n]
    LAB=mnist.valid_lab[:n]
    (cost,valid_per,correct)=ann.g[-1](IMG,params,LAB,1)
    print('valid percent is : %.2f%%'%(valid_per*100))
    return (valid_per,correct) 
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
#with open('p2.pkl', 'wb') as f: pickle.dump(params,f)
with open('p3.pkl', 'rb') as f: params_saved=pickle.load(f)
nx=28*28; ny=10
params_init={'b0':0*np.ones((nx,1)),'b1':0*np.ones((ny,1)),'w1':0.01*np.ones((ny,nx))}
#params=params_saved
params=params_init
## ==========

## ==========
##  training
## ==========
#params=batch(params,0,0,0,1)
#params=batch(params,100,300)
#params=batch(params,200)
#params=batch(params,200,0,.01,1)
#params=batch(params,3,2,.01,1)
#params=batch(params,1,0,.1,1)
#params=batch(params,30,200,.01,1)
#params=batch(params,10,0,.01,1)
#params=batch(params,0,0,.1,0)
#params=batch(params,1,.01,1)
#(valid_per,loss_avg)=valid(params,3)
#(valid_per,corrent)=valid(params,12)
#(valid_per,correct)=valid(params)

## ==========
n=np.random.randint(mnist.test_num)
x=mnist.test_img[0]
lab=mnist.test_lab[0]
#y=ann.fp(x,params)
#y=np.argmax(y)
k1=ann.slope(x,params,lab)
print('k.keys()=',k1.keys())
#show()
#with open('p3.pkl', 'wb') as f: pickle.dump(params,f)
## ==========

