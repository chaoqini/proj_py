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

#n=1	
#x=mnist.test_img[n]
#lab=mnist.test_lab[n].squeeze()
#x=x.reshape(28,28)
###print(x.reshape(28,28))
#print(x.shape)
#
#xp=np.pad(x,1)
##print(xp.shape)
##exit(0)
#
#t1=x[0:3,0:3]
#t2=x[0:3,4:7]
##t1.append(t2)
##t3=np.append(t1,t2,axis=0)
##t1=np.append(t1,t2,axis=1)
##t1=np.vstack((t1,t2))
#t1=np.c_[t1,t2]
##t1=t1.c_[t2]
##t1=t1.hstake(t1)
#print(t1.shape)
#print(t1)
#
##(h,w)=(10,9)
##k=5
#x=np.arange(h*w).reshape(h,w)
#xp=np.pad(x,int(k/2))
#kfilter=np.ones((k,k)).reshape(1,-1)
##tc2=np.zeros((28-3+1,9,28-3+1))
#tc2=np.zeros((xp.shape[0]-k+1,k*k,xp.shape[-1]-k+1))
##print('tc2.shape=',tc2.shape)
##print('tc2 cut shape=',tc2[0,0:9,0].shape)
##print('xp cut shape=',xp[0:3,0:3].shape)
##print('xp cut2 shape=',xp[0:3,0:3].reshape(-1).shape)
##tc2[0,:,0]=xp[0:3,0:3].reshape(-1)
##tc2[0,:,1]=xp[0:3,1:4].reshape(-1)
##tc2[1,:,0]=xp[1:4,0:3].reshape(-1)
##tc2[1,:,1]=xp[1:4,1:4].reshape(-1)
##for r in range(6-3+1):
##	for c in range(6-+3+1):
#for r in range(xp.shape[0]-k+1):
#	for c in range(xp.shape[1]-k+1):
#		tc2[r,:,c]=xp[r:r+k,c:c+k].reshape(-1)
#print('tc2.shape=',tc2.shape)
#print('xp=\n',xp)
#print('tc2=\n',tc2)
#tc3=kfilter@tc2
#print('tc3.shape=',tc3.shape)
#print('tc3=\n',tc3)
#tc4=tc3.reshape(tc3.shape[0],tc3.shape[-1])
#print('x.shape=',x.shape)
#print('tc4.shape=',tc4.shape)
#print('tc4=\n',tc4)
#
def img2d_convolution(img,kfilter):
	(h,w)=img.shape
	kf=kfilter.shape[0]
	k1r=kfilter.reshape(1,-1)
	xp=np.pad(img,int(k/2))
	cols=np.zeros((xp.shape[0]-k+1,k*k,xp.shape[-1]-k+1))
	for r in range(xp.shape[0]-k+1):
		for c in range(xp.shape[1]-k+1):
			cols[r,:,c]=xp[r:r+k,c:c+k].reshape(-1)
	kcols=k1r@cols
	kcols=kcols.reshape(kcols.shape[0],kcols.shape[-1])
	return kcols

def maxpooling(img,k=2):
	(h,w)=img.shape
	(h,w)=(int(h/k),int(w/k))
	cols=np.zeros((h,k*k,w))
#	print('h=',h)
#	print('w=',w)
	for r in range(h):
		for c in range(w):
			cols[r,:,c]=img[k*r:k*r+k,k*c:k*c+k].reshape(-1)
	maxcols=np.max(cols,axis=1)
	return maxcols

#
#(h,w)=(4,9)
#k=3
#x=np.arange(h*w).reshape(h,w)
#kf=np.ones((k,k))
#y=img2d_convolution(x,kf)
#p=maxpooling(y)
#a=relu(p)

#print('x=\n',x)
#print('y=\n',y)
#print('p=\n',p)
#print('a=\n',a)
#print('x.shape=',x.shape)
#print('y.shape=',y.shape)
#print('p.shape=',p.shape)
#print('a.shape=',a.shape)


#
n=1	
k=3
e=1-8
xin=mnist.test_img[n]
lab=mnist.test_lab[n].squeeze()
xin=xin.reshape(28,28)

kf=np.ones((k,k))
z0=img2d_convolution(xin,kf)
z0=maxpooling(z0)
u0=np.mean(z0)
v0=np.var(z0)
x0=(z0-u0)/(v0+e)**0.5
(gama0,beta0)=(np.ones(x0.shape),np.zeros(x0.shape))
y0=gama0*x0+beta0
a0=relu(y0)
#print('z0=\n',z0)
#print('z0.shape=',z0.shape)
#print('y0=\n',y0)
#print('y0.shape=',y0.shape)
#print('a0=\n',a0)
#print('a0.shape=',a0.shape)
#
#plt.imshow(xin,cmap='gray')
#plt.show()
#plt.imshow(z0,cmap='gray')
#plt.show()
#plt.imshow(x0,cmap='gray')
#plt.show()
#plt.imshow(y0,cmap='gray')
#plt.show()


def init_params(lays=3,kr=3,nk=2,nh=28,nw=28,ny=10,func=0,seed=0):
#		 print('init_params:')
	np.random.seed(seed)
	if func==0:  (func,func_d)=(relu,relu_d)
	(nl,params_init,g,g_d,l2_grad)=([],{},[],[],{})
	nl.append(nx)
	for i in range(lays-1): nl.append(nnode)
	nl.append(ny)
	for i in range(lays):
#		params_init['b'+str(i)]=0
		params_init['k'+str(i)]=np.random.randn(kr,kr)
		params_init['beta'+str(i)]=0
		params_init['gama'+str(i)]=1
#		params_init['beta'+str(i)]=np.ones((nh,nw))*0
#		params_init['gama'+str(i)]=np.random.randn(nh,nw)*1e-3
		l2_grad['d_b'+str(i)]=[]
		l2_grad['d_k'+str(i)]=[]
		l2_grad['d_beta'+str(i)]=[]
		l2_grad['d_gama'+str(i)]=[]
		g.append(func)
		g_d.append(func_d)
	g[-1]=softmax;g.append(cross_entropy)
	params=copy.deepcopy(params_init)
	return (params,params_init,g,g_d,l2_grad)
(params,params_init,g,g_d,l2_grad)=init_params()

def fp(X,params,g,isop=0,e=1e-8):
	if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
	assert(X.ndim==3)
	(l,OP)=(int(len(params)/4),{})
	OP['A-1']=X
	l=2
	for i in range(l) :
		ki=params['k'+str(i)]
		gamai=params['gama'+str(i)]
		betai=params['beta'+str(i)]
		Ai_1=OP['A'+str(i-1)]
		Zi=img2d_convolution(Ai_1,ki)
		ui=np.mean(Zi,axis=(1,2),keepdims=1)
		vi=np.var(Zi,axis=(1,2),keepdims=1)
		Xi=(Zi-ui)/(vi+e)**0.5
		Yi=gamai*Xi+betai
		Ai=g[i](Yi)
		OP['Z'+str(i)]=Zi
		OP['X'+str(i)]=Xi
		OP['Y'+str(i)]=Yi
		OP['A'+str(i)]=Ai
		OP['u'+str(i)]=ui
		OP['v'+str(i)]=vi
	Y=OP['A'+str(l-1)]
	if isop==0: 
		return Y
	else: 
		return (Y,OP)


