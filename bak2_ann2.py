#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
from matplotlib import pyplot as plt 
import pickle


def tanh(x): return np.tanh(x)
def d_tanh(x):
    y=tanh(x)
    return 1-y**2
def relu(x): return np.maximum(0,x)
def d_relu(x): return (np.sign(x)+1)/2
def softmax(x): 
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def d_softmax(x): 
    y=softmax(x)
    return np.diag(y)-np.outer(y,y)

class ann:
    logdv=1e-23
    def fp(x,params):
        x=x.reshape(-1,1)
        params['b0']=params['b0'].reshape(-1,1)
        params['b1']=params['b1'].reshape(-1,1)
        b0=params['b0']
        b1=params['b1']
        w1=params['w1']
        z0=x+b0
#        a0=relu(z0)
        a0=tanh(z0)
        z1=w1@a0+b1
        a1=softmax(z1)
        y=a1
#        op={'params':params,'z0':z0,'a0':a0,'z1':z1,'a1':a1}
        op={'z0':z0,'a0':a0,'z1':z1,'a1':a1}
        return (y,op)

    def bp(op,params,lab):
        z0=op['z0'].reshape(-1,1)
        a0=op['a0'].reshape(-1,1)
        z1=op['z1'].reshape(-1,1)
        a1=op['a1'].reshape(-1,1)
        w1=params['w1']
        y=a1
        yl=np.eye(y.shape[0])[lab].reshape(-1,1)
#        d_a1=(1-yl)/(1-y+ann.logdv)-yl/(y+ann.logdv)
        d_a1=2*(y-yl)
        d_z1=d_softmax(z1).T@d_a1
#        d_z0=(w1.T@d_z1)*d_relu(z0)
        d_z0=(w1.T@d_z1)*d_tanh(z0)
        d_w1=d_z1@a0.T
        d_b1=d_z1
        d_b0=d_z0
        grad={'d_w1':d_w1,'d_b1':d_b1,'d_b0':d_b0}
        return grad

    def k(x,params,lab,h=1e-6):
        slope={}
        pp = copy.deepcopy(params) 
        for (k,v) in pp.items():
            slope[k]=np.zeros(v.shape)
            for i in range(len(v)):
                for j in range(len(v[i])):
                    v[i][j]-=h
                    l1=ann.cost(x,pp,lab)
                    v[i][j]+=2*h
                    l2=ann.cost(x,pp,lab)
                    v[i][j]-=h
                    slp=(l2-l1)/(2*h)
                    slope[k][i,j]=slp
        slope['d_b0']=slope.pop('b0')            
        slope['d_b1']=slope.pop('b1')            
        slope['d_w1']=slope.pop('w1')            
        return slope

    def loss(x,parms,lab):
        y=ann.fp(x,params)[0]
        yl=np.eye(y.shape[0])[lab].reshape(-1,1)
        l=(y-yl).T@(y-yl)
        l=np.sum((y-yl)*(y-yl))
        return l

    def cost(x,params,lab):
        y=ann.fp(x,params)[0].reshape(-1,1)
        yl=np.eye(y.shape[0])[lab].reshape(-1,1)
        loss = -yl*np.log(y+ann.logdv)-(1-yl)*np.log(1-y+ann.logdv)
#        m=y.shape[1]
        m=1
        cost=np.sum(loss)/m
        return cost

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


#with open('p2.pkl', 'wb') as f: pickle.dump(params, f)
#with open('p2.pkl', 'rb') as f: params=pickle.load(f)
nx=28*28; ny=10
params_init={'b0':0.1*np.ones(nx),'b1':0.1*np.ones(ny),'w1':1*np.ones([ny,nx])}
params=params_init

#x=np.arange(28*28)*1e-8
#num=11
#x=mnist.test_img[num]
#lab=mnist.test_lab[num]
#img=x.reshape(28,28)
#plt.imshow(img,cmap='gray');plt.show()
#(y,op)=ann.fp(x,params)
#loss=ann.cost(x,params,lab)
#grad=ann.bp(op,params,lab)
#k=ann.k(x,params,lab)

batch_size=100

for n in range(int(mnist.train_num*1e-2)):
    x=mnist.test_img[n]
    lab=mnist.test_lab[n]
    l1=ann.loss(x,params,lab)
    print('='*10)
    print('the loss in %s/%s before update is : %s'%(n,mnist.train_num,l1))
    (y,op)=ann.fp(x,params)
#    grad=ann.bp(op,params,lab)
#    print(grad['d_b1'].T)
    grad=ann.k(x,params,lab)
#    print(grad['b1'].T)
    params=ann.update_params(params,grad,1e3)
    l2=ann.loss(x,params,lab)
    print('the loss in %s/%s after update is : %s'%(n,mnist.train_num,l2))
    print('loss grade is : %s'%(l1-l2))


#params=ann.update_params(params,grad,1)
#print(params.values())
#print(params.keys())
#print(params[params.keys()[0]])
#print(params['b1'].shape)
#print(op['params']['w1'].shape)


#print('='*10)
#print('grad is:')
#print(grad['d_b0'])
#print(grad['d_b1'])
#print(grad['d_w1'])
#print(grad['d_w1'].shape)
#print('='*10)
#print('slope is:')
#print(k['b0'])
#print(k['b1'])
#print(k['w1'])
#print(k['w1'].shape)
#print(k['b0'].shape)
#print('='*10)
#print(k.keys())




#print(d_b1)
#print('cost is : %s'%loss)
#print(ys.shape)
#print(y.shape)
#print(x)
#print(grad['d_b1'])
#print(grad['d_b0'])
#print(grad['d_w1'])
#print(params['b0'].shape)
#print(params['b1'].shape)
#print(params['w1'].shape)
#print(params['b1'])

#print(ys)
#print(ys.shape)
#print(y)
#print(y.shape)
#print(cost)





