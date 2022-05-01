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
def cross_entropy(X,LAB,params,g,isvalid=0):
	Y=bnn.fp(X,params,g)
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

#def im2col():

n=1	
x=mnist.test_img[n]
lab=mnist.test_lab[n].squeeze()
x=x.reshape(28,28)
##print(x.reshape(28,28))
print(x.shape)

xp=np.pad(x,1)
#print(xp.shape)
#exit(0)

t1=x[0:3,0:3]
t2=x[0:3,4:7]
#t1.append(t2)
#t3=np.append(t1,t2,axis=0)
#t1=np.append(t1,t2,axis=1)
#t1=np.vstack((t1,t2))
t1=np.c_[t1,t2]
#t1=t1.c_[t2]
#t1=t1.hstake(t1)
print(t1.shape)
print(t1)

(h,w)=(10,9)
k=5
x=np.arange(h*w).reshape(h,w)
xp=np.pad(x,int(k/2))
kfilter=np.ones((k,k)).reshape(1,-1)
#tc2=np.zeros((28-3+1,9,28-3+1))
tc2=np.zeros((xp.shape[0]-k+1,k*k,xp.shape[-1]-k+1))
#print('tc2.shape=',tc2.shape)
#print('tc2 cut shape=',tc2[0,0:9,0].shape)
#print('xp cut shape=',xp[0:3,0:3].shape)
#print('xp cut2 shape=',xp[0:3,0:3].reshape(-1).shape)
#tc2[0,:,0]=xp[0:3,0:3].reshape(-1)
#tc2[0,:,1]=xp[0:3,1:4].reshape(-1)
#tc2[1,:,0]=xp[1:4,0:3].reshape(-1)
#tc2[1,:,1]=xp[1:4,1:4].reshape(-1)
#for r in range(6-3+1):
#	for c in range(6-+3+1):
for r in range(xp.shape[0]-k+1):
	for c in range(xp.shape[1]-k+1):
		tc2[r,:,c]=xp[r:r+k,c:c+k].reshape(-1)
print('tc2.shape=',tc2.shape)
print('xp=\n',xp)
print('tc2=\n',tc2)
tc3=kfilter@tc2
print('tc3.shape=',tc3.shape)
print('tc3=\n',tc3)
tc4=tc3.reshape(tc3.shape[0],tc3.shape[-1])
print('x.shape=',x.shape)
print('tc4.shape=',tc4.shape)
print('tc4=\n',tc4)







