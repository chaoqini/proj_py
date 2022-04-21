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
def relu(x,kn=0):
    y=copy.deepcopy(x).reshape(-1)
    yt=x.reshape(-1)
    y[yt<0]=y[yt<0]*kn
    y[yt>=0]=y[yt>=0]
    y=y.reshape(x.shape)
    return y
def relu_d(x,kn=0):
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
#    y=dnn.fp(x,params)
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
    Y=dnn.fp(X,params)
#    print('log_loss: Y.shape=',Y.shape)
    meye=np.array([np.eye(Y.shape[1])]*len(LAB))
    lab=LAB.reshape(-1)
    nbatch=np.arange(len(meye))
    YL=meye[nbatch,lab,:]
    YL=YL.reshape(YL.shape+tuple([1]))
#    print('log_loss: Y.shape=',Y.shape)
#    print('log_loss: YL.shape=',YL.shape)
#    print('log_loss: X=',X.transpose(0,2,1))
#    print('log_loss: lab=',lab.T)
#    print('log_loss: Y=',Y.transpose(0,2,1))
#    print('log_loss: YL=',YL.transpose(0,2,1))
#    LOSS=-YL*np.log(Y)
    LOSS=-np.sum(YL*np.log(Y),axis=1)
#    print('log_loss: LOSS.shape=',LOSS.shape)
#    LOSS=np.sum(LOSS,axis=1)
#    print('log_loss: LOSS.shape=',LOSS.shape)
#    print('log_loss: LOSS=',LOSS.T)
    cost=np.sum(LOSS)/len(LOSS)
    if isvalid==0:
        return cost
    else:
#        print('log_loss: Y=',Y.transpose(0,2,1))
        y1d_max=np.max(Y,axis=1,keepdims=1)
#        print('log_loss: y1d_max.shape=',y1d_max.shape)
        Y=np.trunc(Y/y1d_max)
        cmp=Y==YL
        correct=np.trunc(np.sum(cmp,axis=1)/cmp.shape[1])
        valid_per=correct.sum()/len(correct)
#        print('log_loss: correct.shape=',correct.shape)
#        print('log_loss: correct=',correct.T)
        return (cost,valid_per,correct)


## ==========
## ==========
#class dnn(lri=0.1,batchi=10):
class dnn:
#    g=(tanh,relu,softmax,log_loss);g_d=(tanh_d,relu_d,softmax_d,log_loss_d)
#    g=(tanh,tanh,softmax,log_loss);g_d=(tanh_d,tanh_d,None,None)
#    g=(tanh,relu,softmax,log_loss);g_d=(tanh_d,relu_d,None,None)
    g=(relu,relu,softmax,log_loss);g_d=(relu_d,relu_d,None,None)
    (lr,batch)=(0.1,20)
#    def __init__(self,lr=0.06,batch=16):
#        self.g=(relu,relu,softmax,log_loss);self.g_d=(relu_d,relu_d,None,None)
#        (self.lr,self.batch)=(lr,batch)

    def fp(X,params,isop=0):
        if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
        assert(X.ndim==3)
        b0=params['b0'].reshape(-1,1)
        b1=params['b1'].reshape(-1,1)
        b2=params['b2'].reshape(-1,1)
        w0=params['w0']
        w1=params['w1']
        w2=params['w2']
        Z0=w0@X+b0
        A0=dnn.g[0](Z0)
        Z1=w1@A0+b1
        A1=dnn.g[1](Z1)
        Z2=w2@A1+b2
        A2=dnn.g[2](Z2)
        Y=A2
#        print('fp: Y=',Y)
        if isop==0:
            return Y
        else:
#            OP={'Z0':Z0,'A0':A0,'Z1':Z1,'A1':A1}
            OP={'Z':[Z0,Z1,Z2],'A':[A0,A1,A2]}
            return (Y,OP)

## ==========
    def bp(X,params,LAB):
        if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
        if LAB.ndim==2: LAB=LAB.reshape(tuple([1])+LAB.shape)
        assert(X.ndim==3)
        assert(LAB.ndim==3)
        (Y,OP)=dnn.fp(X,params,1)
        assert(Y.ndim==3)
        Z0=OP['Z'][0]
        Z1=OP['Z'][1]
        Z2=OP['Z'][2]
        A0=OP['A'][0]
        A1=OP['A'][1]
        A2=OP['A'][2]
        w0=params['w0']
        w1=params['w1']
        w2=params['w2']
        meye=np.array([np.eye(Y.shape[1])]*len(LAB))
        assert(meye.ndim==3)
        lab=LAB.reshape(-1)
        assert(lab.ndim==1)
        nbatch=np.arange(len(meye))
        YL=meye[nbatch,lab,:]
        YL=YL.reshape(YL.shape+(1,))
        d_Y=Y-YL
        d_Z2=d_Y
        d_Z1=dnn.g_d[1](Z1)*(w2.T@d_Z2)
        d_Z0=dnn.g_d[0](Z0)*(w1.T@d_Z1)
        d_W2=d_Z2@A1.transpose(0,2,1)
        d_W1=d_Z1@A0.transpose(0,2,1)
        d_W0=d_Z0@X.transpose(0,2,1)
        d_B2=d_Z2
        d_B1=d_Z1
        d_B0=d_Z0
        d_b0=np.mean(d_B0,axis=0)
        d_b1=np.mean(d_B1,axis=0)
        d_b2=np.mean(d_B2,axis=0)
        d_w0=np.mean(d_W0,axis=0)
        d_w1=np.mean(d_W1,axis=0)
        d_w2=np.mean(d_W2,axis=0)
        grad={'d_b0':d_b0,'d_b1':d_b1,'d_b2':d_b2,'d_w0':d_w0,'d_w1':d_w1,'d_w2':d_w2}
#        grad={'d_w':[0,d_w1,d_w2],'d_b':[d_b0,d_b1,d_b2]}
        return grad

## ==========
    def slope(x,params,lab,dv=1e-5):
#        print('slope:')
        slp={}
        pt=copy.deepcopy(params) 
        for (k,v) in pt.items():
            d_k='d_'+k
            slp[d_k]=np.zeros(v.shape)
            for i in range(len(v)):
                for j in range(len(v[i])):
                    vb=v[i,j]
                    v[i,j]=vb-dv
                    l1=dnn.g[-1](x,pt,lab)
                    v[i,j]=vb+dv
                    l2=dnn.g[-1](x,pt,lab)
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
        y1=dnn.bp(x,params,lab)
        y2=dnn.slope(x,params,lab,dv)
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
    def normalize(X):
        AX=mnist.train_img
        mean=AX.mean()
        std=AX.std()
        return (X-mean)/std

## ==========
#mnist.train_num=50000
def batch(params,lr=0,batch=0,batches=0,isplot=0,istime=0):
    if lr==0: lr=dnn.lr
    if batch<1: batch=dnn.batch
    max_batches=int(len(mnist.train_img)/batch)
    if batches<1: batches=max_batches
    batches=min(max_batches,int(batches))
    X=mnist.train_img[:batch*batches]
    LAB=mnist.train_lab[:batch*batches]
    X=X.reshape((-1,batch)+X.shape[1:3])
    X=dnn.normalize(X)
    LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
    print('Train learning rate =',lr)
    print('Train batch =',batch)
    print('Train input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
    for k in params.keys():
        print( 'params %s.shape='%k,params[k].shape)
    for i in range(len(dnn.g)):
        print('active function g[%d] is:'%i,dnn.g[i].__name__)
    cost=[]
    print('Training bath running ...')
    for i in range(len(X)):
        pn=i%(max(int(len(X)/10),1))
        if pn==0 or i==len(X)-1:
            print('Training iteration number = %s/%s'%(i,len(X)))
            if istime!=0:tb=time.time()
        grad=dnn.bp(X[i],params,LAB[i])
        params=dnn.update_params(params,grad,lr)
        (cost_i,valid_per,correct)=dnn.g[-1](X[i],params,LAB[i],1)
        cost.append(cost_i)
        if (pn==0 or i==len(X)-1) and istime!=0:
            te=time.time()
            tspd=(te-tb)*1000
            print('the spending time of %s/%s batch is %s mS'%(i,len(X),tspd))
#        print('batch: cost_i=',cost_i)
    cost=np.array(cost)
#    cost=np.array(cost)[50:-1]
    if isplot!=0:
        plt.plot(cost)
        plt.ylabel('Cost')
        plt.xlabel('Iterations *%s'%batch)
        var_title=(dnn.g[0].__name__,dnn.g[1].__name__,dnn.g[2].__name__,lr)
        title='Active g[0]= %s\n Active g[1]= %s\n Active g[2]= %s\n \
        Learning rate = %s\n'%var_title
        plt.title(title)
        plt.show()
    return params

## ==========
def valid(params,batch=0,batches=0):
    if batch<1: batch=100
    max_batches=int(len(mnist.valid_img)/batch)
    if batches<1: batches=max_batches
    batches=min(max_batches,int(batches))
    X=mnist.valid_img[:batch*batches]
    LAB=mnist.valid_lab[:batch*batches]
    X=X.reshape((-1,batch)+X.shape[1:3])
    X=dnn.normalize(X)
    LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
    print('Valid Input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
    (cost,valid_per,correct)=([],[],[])
    print('Valid batch running ...')
    for i in range(len(X)):
        pn=i%(max(int(len(X)/10),1))
        if pn==0 or i==len(X)-1:
            print('Valid iteration number = %s/%s'%(i,len(X)))
        (cost_i,valid_per_i,correct_i)=dnn.g[-1](X[i],params,LAB[i],1)
        cost.append(cost_i)
        valid_per.append(valid_per_i)
        correct.append(correct_i)
#        print('Valid: cost_i=',cost_i)
#        print('Valid: valid_per_i=',valid_per_i)
#        print('Valid: correct_i=',correct_i)
    cost=np.array(cost)
    valid_per=np.array(valid_per)
    correct=np.array(correct)
#    print('valid: cost.shape=',cost.shape)
#    print('valid: valid_per.shape=',valid_per.shape)
#    print('valid: correct.shape=',correct.shape)
    print('Valid L2 norm:')
    valid_per=valid_per.sum()/len(valid_per)
    for (k,v) in params.items():
        L2=np.linalg.norm(v)/v.size
        print('Valid L2_normalize_%s = %s'%(k,L2))
    print('Valid percent is : %.2f%%'%(valid_per*100))
    return (valid_per,correct) 
## ==========
def valid_train(params,batch=0,batches=0):
    if batch<1: batch=100
    max_batches=int(len(mnist.train_img)/batch)
    if batches<1: batches=max_batches
    batches=min(max_batches,int(batches))
    X=mnist.train_img[:batch*batches]
    LAB=mnist.train_lab[:batch*batches]
    X=X.reshape((-1,batch)+X.shape[1:3])
    X=dnn.normalize(X)
    LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
    print('Valid_train Input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
    (cost,valid_per,correct)=([],[],[])
    print('Valid_train batch running ...')
    for i in range(len(X)):
        pn=i%(max(int(len(X)/10),1))
        if pn==0 or i==len(X)-1:
            print('Valid_train iteration number = %s/%s'%(i,len(X)))
        (cost_i,valid_per_i,correct_i)=dnn.g[-1](X[i],params,LAB[i],1)
        cost.append(cost_i)
        valid_per.append(valid_per_i)
        correct.append(correct_i)
    cost=np.array(cost)
    valid_per=np.array(valid_per)
    correct=np.array(correct)
#    print('Valid_train: cost.shape=',cost.shape)
#    print('Valid_train: valid_per.shape=',valid_per.shape)
#    print('Valid_train: correct.shape=',correct.shape)
#    print('Valid_train L2 norm:')
    valid_per=valid_per.sum()/len(valid_per)
    for (k,v) in params.items():
        L2=np.linalg.norm(v)/v.size
        print('Valid_train L2_normalize_%s = %s'%(k,L2))
    print('Valid_train percent is : %.2f%%'%(valid_per*100))
    return (valid_per,correct) 
## ==========
## ==========
def show(n=-1):
    if n==-1: n=np.random.randint(mnist.test_num)
    x=mnist.test_img[n]
    lab=mnist.test_lab[n].squeeze()
    y=dnn.fp(x,params)
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
with open('dnn_p1.pkl', 'rb') as f: params_saved=pickle.load(f)
#(nx,nz0,nz1,ny)=(28*28,784,100,10)
(nx,nz0,nz1,ny)=(28*28,400,100,10)
#params_init={'b0':0*np.ones((nx,1)),'b1':0*np.ones((ny,1)),'w1':0.01*np.ones((ny,nx))}
np.random.seed(1)
b0=np.ones((nz0,1))*0
b1=np.ones((nz1,1))*0
b2=np.ones((ny,1))*0
w0=np.random.randn(nz0,nx)*1e-3
w1=np.random.randn(nz1,nz0)*1e-3
w2=np.random.randn(ny,nz1)*1e-3
params_init={'b0':b0,'b1':b1,'b2':b2,'w0':w0,'w1':w1,'w2':w2}
#params_init={'b0':1e-3*np.ones((nx,1)),'b1':1e-3*np.ones((ny,1)),'w1':0.01*np.ones((ny,nx))}
#print('params_init[b0]=',params_init['b0'])
#params=params_saved
params=params_init
## ==========

## ==========
##  training
print('Training running ...')
## ==========
#params=batch(params,0,20,0,1,1)
#params=batch(params,0.1,100,0,1,1)
#params=batch(params,0.1,20,0,1,1)
params=batch(params,0,0,0,1,1)
## ==========
## valid
print('Valid running ...')
## ==========
#(valid_per,correct)=valid(params,3,2)
(valid_per,correct)=valid(params)
(valid_per2,correct2)=valid_train(params)
## ==========

## ==========
# ==========
with open('dnn_p1.pkl','wb') as f: pickle.dump(params,f);print('params write in %s'%f.name)
## ==========
## ==========
## grade check
print('Grade check running ...')
## ==========
num=np.random.randint(mnist.test_num)
num=11
x=mnist.test_img[num]
lab=mnist.test_lab[num]
#y=dnn.fp(x,params)
#y=np.argmax(y)
#k1=dnn.slope(x,params,lab)
dnn.grad_check(x,params,lab)
print('Grade check end.')
#show()

