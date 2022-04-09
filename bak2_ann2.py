#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
from matplotlib import pyplot as plt 
import pickle


#def relu(x): return np.max(0,x)
#def relu(x,k=0): return np.maximum(k*x,x)
#def relu(x,k=0): return max(k,x)
#def relu(x,k=0):k=(k if k>0 and k<1 else 0);return np.max(k*x,x)

def relu(x): return np.maximum(0,x)
#def d_relu(x): return (1 if x>0 else 0)
def d_relu(x): return (np.sign(x)+1)/2
def softmax(x): 
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def d_softmax(x): 
    y=softmax(x)
    return np.diag(y)-np.outer(y,y)

class ann:
#    def __init__(self): pass
    def fp(x,params):
        x=x.reshape(-1,1)
        b0=params['b0'].reshape(-1,1)
        b1=params['b1'].reshape(-1,1)
        w1=params['w1'].reshape(b1.shape[0],-1)
        z0=x+b0
        a0=relu(z0)
        z1=w1@a0+b1
        a1=softmax(z1)
        ys=a1
        cache={'params':params,'z0':z0,'a0':a0,'z1':z1,'a1':a1}
#        print(z0)
#        print(z0.shape)
#        print(x.shape)
#        print(a0)
#        print(a0.shape)
#        print(z1)
#        print(z1.shape)
#        print(ys)
#        print(ys.shape)
        return (ys,cache)
    def bp(cache,y):

        y=y.reshape(-1,1)
        z0=cache['z0'].reshape(-1,1)
        a0=cache['a0'].reshape(-1,1)
        z1=cache['z1'].reshape(-1,1)
        a1=cache['a1'].reshape(-1,1)
        w1=cache['params']['w1'].reshape(z1.shape[0],-1)

        d_a1=(1-y)/(1-a1)-y/a1
        d_z1=d_softmax(z1).T@d_a1
        d_z0=(w1.T@d_z1)*d_relu(z0)

        d_w1=d_z1@a0.T
        d_b1=d_z1
        d_b0=d_z0

        grad={'d_w1':d_w1,'d_b1':d_b1,'d_b0':d_b0}
        return grad
#        print(w1.shape)
#        print(d_a1.shape)
#        print(y.shape)
#        print(a1.shape)
#        print(d_a1.shape)
#        print(d_z1.shape)
#        print(d_z0.shape)
#        print(a0.T.shape)
#        print(d_w1.shape)
#        print(d_b1.shape)
#        print(d_b0.shape)



    def cost(ys,y):
        logprobs = y*np.log(ys)+(1-y)*np.log(1-ys)
#        m=y.shape[1]
        m=1
        cost=np.sum(logprobs)/m
#        x1=y*np.log(ys)
#        x2=(1-y)*np.log(1-ys)
#        x=np.sum(x1+x2)
#        print(ys)
#        print(y)
#        print(x1+x2)
#        print(x)

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


        



with open('p2.pkl', 'rb') as f: params=pickle.load(f)
#print(params.keys())
#params['w1']=params.pop('w')
#print(params.keys())
#with open('p2.pkl', 'wb') as f: pickle.dump(params, f)

x=np.arange(28*28)*1e-8
(ys,cache)=ann.fp(x,params)
y=np.eye(ys.shape[0])[3]
cost=ann.cost(ys,y)
grad=ann.bp(cache,y)
params=ann.update_params(params,grad,1)


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





