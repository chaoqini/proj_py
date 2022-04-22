##!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
from matplotlib import pyplot as plt 
import pickle
import sys
import time
import copy


## ==========
def tanh(x): return np.tanh(x)
def tanh_d(x):
    y=tanh(x)
    return 1-y**2
#def relu(x): return np.maximum(0,x)
def relu(x,kn=0):
    y=x.copy().reshape(-1)
    yt=x.reshape(-1)
    y[yt<0]=y[yt<0]*kn
    y[yt>=0]=y[yt>=0]
    y=y.reshape(x.shape)
    return y
def relu_d(x,kn=0):
    y=x.copy().reshape(-1)
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
def log_loss(X,params,LAB,isvalid=0):
    Y=dnn.fp(X,params)
#    print('log_loss: Y.shape=',Y.shape)
    meye=np.array([np.eye(Y.shape[1])]*len(LAB))
    lab=LAB.reshape(-1)
    nbatch=np.arange(len(meye))
    YL=meye[nbatch,lab,:]
    YL=YL.reshape(YL.shape+tuple([1]))
    LOSS=-np.sum(YL*np.log(Y),axis=1)
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
class dnn:
    def init_dnn(lays=0,nx=0,ny=0,nn=0,f=0,seed=0,lr0=0,klr=0,batch=0):
        np.random.seed(seed)
        if lays==0: lays=3
        if nx==0: nx=28*28
        if ny==0: ny=10
        if nn==0: nn=100
        if f==0:  (f,f_d)=(relu,relu_d)
        if lr0==0:  lr0=0.1
        if klr==0:  klr=1
        if batch==0:  batch=20
        (nl,params_init,g,g_d)=([nx],{},[],[])
        for i in range(lays): nl.append(nn)
        nl.append(ny)
        for i in range(lays):
            bi=('b'+str(i))
            wi=('w'+str(i))
            params_init[bi]=np.ones((nl[i+1],1))*0
            params_init[wi]=np.random.randn(nl[i+1],nl[i])*1e-3
            g.append(f)
            g_d.append(f_d)
        g[-1]=softmax
#        g.append(softmax)
        g.append(log_loss)
        params=copy.deepcopy(params_init)
        (lr0,klr,batch)=(0.1,1,20)
        return (params,lays,g,g_d,lr0,klr,batch)
    (params,lays,g,g_d,lr0,klr,batch)=init_dnn()

## ==========
    def fp(X,params,isop=0):
        if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
        assert(X.ndim==3)
        (l,OP,p)=(int(len(params)/2),{},params)
        OP['A-1']=X
        for i in range(l):
            OP['Z'+str(i)]=p['w'+str(i)]@OP['A'+str(i-1)]+p['b'+str(i)]
            OP['A'+str(i)]=dnn.g[i](OP['Z'+str(i)])
        Y=OP['A'+str(l-1)]
        if isop==0: 
            return Y
        else: 
            return (Y,OP)
## ==========
    def bp(X,params,LAB):
        if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
        if LAB.ndim==2: LAB=LAB.reshape(tuple([1])+LAB.shape)
        assert(X.ndim==3); assert(LAB.ndim==3)
        (Y,OP)=dnn.fp(X,params,1)
        meye=np.array([np.eye(Y.shape[1])]*len(LAB))
        lab=LAB.reshape(-1)
        assert(Y.ndim==3);assert(lab.ndim==1);assert(meye.ndim==3);
        nbatch=np.arange(len(meye))
        YL=meye[nbatch,lab,:]
        YL=YL.reshape(YL.shape+tuple([1]))
        (l,d_,grad)=(int(len(params)/2),{},{})
        for i in range(l-1,-1,-1):
            Zi=('Z'+str(i))
            Zip1=('Z'+str(i+1))
            Ai=('A'+str(i))
            Ain1=('A'+str(i-1))
            wi=('w'+str(i))
            wip1=('w'+str(i+1))
            bi=('b'+str(i))
            if i==l-1: d_[Zi]=Y-YL
            else: d_[Zi]=dnn.g_d[i](OP[Zi])*(params[wip1].T@d_[Zip1])
            d_[wi]=d_[Zi]@OP[Ain1].transpose(0,2,1)
            d_[bi]=d_[Zi]
            grad['d_'+wi]=np.mean(d_[wi],axis=0)
            grad['d_'+bi]=np.mean(d_[bi],axis=0)
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
## ==========
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
    def update_params(params,d,lr=1):
        for k in params.keys():
            params[k]-=lr*d['d_'+k]
        return params
    def update_params_adam(params,grad,lr,v,s,t=1):
        (beta1,beta2,e)=(0.9,0.999,1e-8)
        for k in params.keys():
#            v['v_'+k] = beta1*v['v_'+k]+(1-beta1)*d['d_'+k]
#            s['s_'+k] = beta2*s['s_'+k]+(1-beta2)*d['d_'+k]**2
#            vc_k = v['v_'+k]/(1-beta1**t)
#            sc_k = s['s_'+k]/(1-beta2**t)
            v[k] = beta1*v[k]+(1-beta1)*grad['d_'+k]
            s[k] = beta2*s[k]+(1-beta2)*grad['d_'+k]**2
            vc_k = v[k]/(1-beta1**t)
            sc_k = s[k]/(1-beta2**t)
            adam_d_k = vc_k/(sc_k**0.5+e)
            params[k]-=lr*adam_d_k
        return (params,v,s)
    def init_adam(params):
        (v,s)=({},{})
        for k in params.keys():
            v[k] = np.zeros(params[k].shape)
            s[k] = np.zeros(params[k].shape)
        return (v,s)
## ==========
    def normalize(X):
#        AX=mnist.train_img
        AX=X
        mean=AX.mean()
        std=AX.std()
        return (X-mean)/std
## ==========
#mnist.train_num=50000
def batch_train(params,lr0=0,klr=0,batch=0,batches=0,isplot=0,istime=0):
    if lr0==0: lr0=dnn.lr0
    if klr==0: klr=dnn.klr
    if batch<1: batch=dnn.batch
    max_batches=int(len(mnist.train_img)/batch)
    if batches<1: batches=max_batches
    batches=min(max_batches,int(batches))
    X=mnist.train_img[:batch*batches]
    LAB=mnist.train_lab[:batch*batches]
    X=X.reshape((-1,batch)+X.shape[1:3])
    X=dnn.normalize(X)
    LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
    print('Train layers =',dnn.lays)
    print('Train learning rate lr0 =',lr0)
    print('Train learning rate klr =',klr)
    print('Train batch =',batch)
    print('Train input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
    for k in params.keys():
        print( 'params %s.shape='%k,params[k].shape)
    for i in range(len(dnn.g)):
        print('active function g[%d] is:'%i,dnn.g[i].__name__)
    (cost,lra)=([],[])
    print('Training bath running ...')
    for i in range(len(X)):
        pn=i%(max(int(len(X)/10),1))
        if pn==0 or i==len(X)-1:
            print('Training iteration number = %s/%s'%(i,len(X)))
            if istime!=0:tb=time.time()
        grad=dnn.bp(X[i],params,LAB[i])
#        lr=lr0/(1+i/100)
        lr=lr0*klr**i
        lra.append(lr)
#        params=dnn.update_params(params,grad,lr)
        if i==0: (v,s)=dnn.init_adam(params)
        else: (params,v,s)=dnn.update_params_adam(params,grad,lr,v,s,i)
        (cost_i,valid_per,correct)=dnn.g[-1](X[i],params,LAB[i],1)
        cost.append(cost_i)
        if (pn==0 or i==len(X)-1) and istime!=0:
            te=time.time()
            tspd=(te-tb)*1000
            print('the spending time of %s/%s batch is %s mS'%(i,len(X),tspd))
#        print('batch: cost_i=',cost_i)
    cost=np.array(cost)
    lra=np.array(lra)
#    cost=np.array(cost)[50:-1]
    if isplot!=0:
        plt.figure()
        plt.subplot(211)
        plt.plot(cost)
        plt.ylabel('Cost')
        plt.subplot(212)
        plt.plot(lra)
        plt.ylabel('lra')
        plt.xlabel('Iterations *%s'%batch)
        var_title=(dnn.g[0].__name__,dnn.g[1].__name__,dnn.g[2].__name__,lr0,klr,batch)
        title='Active g[0]=%s\n Active g[1]=%s\n Active g[2]=%s\n \
        lr0=%s\n klr=%s\n batch=%s\n'%var_title
        plt.title(title)
#        leg=plt.legend(title=title)
#        leg._legend_box.align="left"
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



def dnn_train(params,lr0=0,klr=0,batch=0,batches=0,isplot=0,istime=0,ischeck=0):
#    params=params_ref.copy()
    params=copy.deepcopy(params)
#    iseq=1
#    for k in params.keys():
#        iseq=iseq&(np.all(params[k]==params_init[k])) 
#        print('params[%s] iseq=%s'%(k,iseq))
    print('Training running ...')
    params=batch_train(params,lr0,klr,batch,batches,isplot,istime)
    print('Valid running ...')
    (valid_per,correct)=valid(params)
    (valid_per2,correct2)=valid_train(params)
    if ischeck==1:
        print('Grade check running ...')
        dnn.grad_check(x,params,lab)
        print('Grade check end.')
    return (valid_per,valid_per2)

#return (params,lays,g,g_d,lr0,klr,batch)
params=dnn.params

def hyperparams_test(params,n=0,lr0=0,klr=0,batch=0,batches=0):
    if n==0: n=5
    (v1a,v2a)=([],[])
    for i in range(n):
        print('hyperparams_test runing iteration=%s/%s'%(i+1,n))
#        (v1,v2)=dnn_train(params,lr0,klr,batch,batches,1)
        j=10**(-i/10*1)*lr0
        (v1,v2)=dnn_train(params,j,0.998,batch,batches,1)
#        (v1,v2)=dnn_train(params,0.9,1-0.001*i,batch,batches,1)
        v1a.append(v1)
        v2a.append(v2)
    v1a=np.array(v1a)
    v2a=np.array(v2a)
    plt.figure()
    plt.subplot(211)
    plt.plot(v1a)
    plt.ylabel('v1a')
    plt.subplot(212)
    plt.plot(v2a)
    plt.ylabel('v2a')
    plt.show()

#hyperparams_test(dnn.params,batch=30,n=10,lr0=0.01)
