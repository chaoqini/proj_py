#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
from matplotlib import pyplot as plt 
import pickle


def relu(x): return np.maximum(0,x)
def d_relu(x): return (np.sign(x)+1)/2
def softmax(x): 
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def d_softmax(x): 
    y=softmax(x)
    return np.diag(y)-np.outer(y,y)

class ann:
    def fp(x,params):
        x=x.reshape(-1,1)
        params['b0']=params['b0'].reshape(-1,1)
        params['b1']=params['b1'].reshape(-1,1)
        b0=params['b0']
        b1=params['b1']
        w1=params['w1']
        z0=x+b0
        a0=relu(z0)
        z1=w1@a0+b1
        a1=softmax(z1)
        ys=a1
        cache={'params':params,'z0':z0,'a0':a0,'z1':z1,'a1':a1}
        return (ys,cache)

    def bp(cache,lab):
        z0=cache['z0'].reshape(-1,1)
        a0=cache['a0'].reshape(-1,1)
        z1=cache['z1'].reshape(-1,1)
        a1=cache['a1'].reshape(-1,1)
        w1=cache['params']['w1']
        ys=a1
        y=np.eye(ys.shape[0])[lab].reshape(-1,1)
        d_a1=(1-y)/(1-a1)-y/a1
        d_z1=d_softmax(z1).T@d_a1
        d_z0=(w1.T@d_z1)*d_relu(z0)
        d_w1=d_z1@a0.T
        d_b1=d_z1
        d_b0=d_z0
        grad={'d_w1':d_w1,'d_b1':d_b1,'d_b0':d_b0}
        return grad

    def k(x,params,lab,h=1e-3):
        slope={}
        params_temp = copy.deepcopy(params) 
        for k in params_temp.keys():
            params_temp[k]-=h
            y1=ann.cost(x,params_temp,lab)
            params_temp[k]+=2*h
            y2=ann.cost(x,params_temp,lab)
            params_temp[k]-=h
            slope[k]=(y2-y1)/(2*h)
        return slope


#       
    def cost(x,params,lab):
        ys=ann.fp(x,params)[0].reshape(-1,1)
        y=np.eye(ys.shape[0])[lab].reshape(-1,1)
        dys=1e-23
        logprobs = y*np.log(ys+dys)+(1-y)*np.log(1-ys+dys)
#        m=y.shape[1]
        m=1
        cost=np.sum(logprobs)/m
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
num=1
x=mnist.test_img[num]
lab=mnist.test_lab[num]
img=x.reshape(28,28)
#plt.imshow(img,cmap='gray');plt.show()
#print(x.shape)
(ys,cache)=ann.fp(x,params)
#ys=ann.fp(x,params)[0]
print(ys.shape)
cost=ann.cost(x,params,lab)
grad=ann.bp(cache,lab)
params=ann.update_params(params,grad,1)
#print(params.values())
#print(params.keys())
#print(params[params.keys()[0]])
#print(params['b1'].shape)
print(cache['params']['w1'].shape)
k=ann.k(x,params,lab)
print('='*10)
print(grad['d_b1'])
print('='*10)
print(k['b1'])
print(k.keys())




#print(d_b1)
print('cost is : %s'%cost)
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





