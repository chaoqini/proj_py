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
#	if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
	assert(X.ndim==4)
	exp=np.exp(X-np.max(X,axis=(1,2,3),keepdims=1))
	expsum=np.sum(exp,axis=(1,2,3),keepdims=1)
	return exp/expsum
def cross_entropy(X,LAB,params,g,isvalid=0):
	assert(X.ndim==4)
	Y=bnn.fp(X,params,g)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
	YL[np.arange(ba),0,LAB.reshape(ba),0]=1
#	meye=np.array([np.eye(Y.shape[1])]*len(LAB))
#	lab=LAB.reshape(-1)
#	nbatch=np.arange(len(meye))
#	YL=meye[nbatch,lab,:]
#	YL=YL.reshape(YL.shape+tuple([1]))
#	LOSS=-np.sum(YL*np.log(Y),axis=(1,2,3))
	LOSS=-np.sum(YL*np.log(Y))
	cost=LOSS/ba
	if isvalid==0:
		return cost
	else:
		y2d_max=np.max(Y,axis=2,keepdims=1)
		Y=np.trunc(Y/y2d_max)
		cmp=Y==YL
		correct=np.trunc(np.sum(cmp,axis=2)/cmp.shape[2])
		valid_per=correct.sum()/len(correct)
		return (cost,valid_per,correct)
## ==========

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
def img_convolution(X,kfilter):
	if X.ndim==2: X=X.reshape((1,)+(X.shape[0],)+(1,)+(X.shape[-1],))
	elif X.ndim==3 and X.shape[-2]!=1: X=X.reshape(X.shape[:-1]+(1,)+X.shape[-1:])
	assert(X.ndim==4)
	kf=kfilter.shape[1]
	p=int(kf/2)
	k1r=kfilter.reshape(1,-1)
	X=np.pad(X,((0,0),(p,p),(0,0),(p,p)))
	(ba,h,Non,w)=X.shape
	(cols_r,cols_c)=(h-kf+1,w-kf+1)
	cols=np.zeros((ba,cols_r,kf*kf,cols_c))
	for r in range(cols_r):
		for c in range(cols_c):
			cols[:,r,:,c]=X[:,r:r+kf,0,c:c+kf].reshape(X.shape[0],-1)
	kcols=k1r@cols
#	kcols=kcols.reshape(kcols.shape[0],kcols.shape[1],kcols.shape[3])
	return kcols

def maxpooling(X,k=2):
	if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
	assert(X.ndim==3)
	(h,w)=X.shape[1:3]
	(h,w)=(int(h/k),int(w/k))
	cols=np.zeros((X.shape[0],h,k*k,w))
#	print('h=',h)
#	print('w=',w)
	for r in range(h):
		for c in range(w):
			cols[:,r,:,c]=X[:,k*r:k*r+k,k*c:k*c+k].reshape(X.shape[0],-1)
	maxcols=np.max(cols,axis=2)
	return maxcols

def init_params(lays=3,kr=3,nk=2,nh=28,nw=28,ny=10,func=0,seed=0):
#		 print('init_params:')
	np.random.seed(seed)
	if func==0:  (func,func_d)=(relu,relu_d)
	(params_init,g,g_d,l2_grad)=({},[],[],{})
	for i in range(lays):
		if i==lays-1:
			params_init['w'+str(i)]=np.random.randn(ny,nh*nw)*1e-3
		else:
			params_init['k'+str(i)]=np.random.randn(kr,kr)
		params_init['gama'+str(i)]=np.array(1.0)
		params_init['beta'+str(i)]=np.array(0.0)
		g.append(func)
		g_d.append(func_d)
#	params_init['w'+str(lays)]=np.random.randn(nh*nw,ny)*1e-3
#	params_init['b'+str(lays)]=np.ones((ny,1))*0
	g[-1]=softmax;g.append(cross_entropy)
	params=copy.deepcopy(params_init)
	return (params,params_init,g,g_d,l2_grad)
(params,params_init,g,g_d,l2_grad)=init_params()

def fp(X,params,g,isop=0,e=1e-8):
	if X.ndim==2: X=X.reshape((1,)+(X.shape[0],)+(1,)+(X.shape[-1],))
	elif X.ndim==3 and X.shape[-2]!=1: X=X.reshape(X.shape[:-1]+(1,)+X.shape[-1:])
	assert(X.ndim==4)
#	if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
#	assert(X.ndim==3)
#	print('X.shape=',X.shape)
	(l,OP)=(int(len(params)/3),{})
	OP['A-1']=X
	for i in range(l) :
		Ai_1=OP['A'+str(i-1)]
		if i==l-1:
			wi=params['w'+str(i)]
			Ai_1=Ai_1.reshape(Ai_1.shape[0],1,-1,1)
			Zi=wi@Ai_1
		else:
			ki=params['k'+str(i)]
			Zi=img_convolution(Ai_1,ki)
		gamai=params['gama'+str(i)]
		betai=params['beta'+str(i)]
		ui=np.mean(Zi,axis=(1,2,3),keepdims=1)
		vi=np.var(Zi,axis=(1,2,3),keepdims=1)
		Xi=(Zi-ui)/(vi+e)**0.5
		Yi=gamai*Xi+betai
		Ai=g[i](Yi)
		OP['Z'+str(i)]=Zi
		OP['X'+str(i)]=Xi
		OP['Y'+str(i)]=Yi
		OP['A'+str(i)]=Ai
		OP['u'+str(i)]=ui
		OP['v'+str(i)]=vi
#	print('Al_1.shape=',Al_1.shape)
	Y=OP['A'+str(l-1)]
	if isop==0: 
		return Y
	else: 
		return (Y,OP)

## ==========
def bp(X,LAB,params,g,g_d,e=1e-8):
	if X.ndim==2: X=X.reshape((1,)+(X.shape[0],)+(1,)+(X.shape[-1],))
	elif X.ndim==3 and X.shape[-2]!=1: X=X.reshape(X.shape[:-1]+(1,)+X.shape[-1:])
	assert(X.ndim==4)
#	if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
	if LAB.ndim==2: LAB=LAB.reshape((1,)+LAB.shape)
	assert(X.ndim==4 and LAB.ndim==3)
	(Y,OP)=fp(X,params,g,isop=1)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
	YL[np.arange(ba),0,LAB.reshape(ba),0]=1
	(l,d_,grad)=(int(len(params)/3),{},{})
	for i in range(l-1,-1,-1):
		if i==l-1:
			wi=params['w'+str(i)]
		else:
			ki=params['k'+str(i)]
		gamai=params['gama'+str(i)]
		betai=params['beta'+str(i)]
		ui=OP['u'+str(i)]
		vi=OP['v'+str(i)]
		Xi=OP['X'+str(i)]
		if i==l-1: d_Yi=Y-YL
		else: d_Yi=d_['Y'+str(i)]
		d_Xi=gamai*d_Yi
#		(batch,mx,nx)=Xi.shape
		Xi=Xi.reshape(Xi.shape[0],-1,1)
		XX=np.einsum('mij,mkj->mik',Xi,Xi)
		Imm=np.ones((XX.shape))
		mmE=np.zeros((XX.shape))
		np.einsum('mii->mi',mmE)[:]=mmE.shape[1]
#		vi=vi.reshape((-1,)+tuple([1]*(XX.ndim-1)))
		dXi_Zi=(mmE-Imm-XX)/(mmE.shape[1]*(vi+e)**0.5)
		d_Zi=dXi_Zi.transpose(0,2,1)@d_Xi
		if i==l-1: 
			d_Ain1=wi.T@d_Zi
		else:
			d_Zi=np.expand_dims(d_Zi,axis=-2)
			ki=ki.reshape(ki.shape[0],1,-1)
			d_Cin1=ki.transpose(0,2,1)@d_Zi
			d_Ain1=d_Cin1
		if i>=1:
			Yin1=OP['Y'+str(i-1)]
			d_Yin1=g_d[i-1](Yin1)*d_Ain1
			d_['Y'+str(i-1)]=d_Yin1
		Ain1=OP['A'+str(i-1)]
		d_wi=d_Zi@Ain1.transpose(0,2,1)
		d_bi=d_Zi
		d_gamai=d_Yi*Xi
		d_betai=d_Yi
		grad['d_w'+str(i)]=d_wi.mean(axis=0)
		grad['d_gama'+str(i)]=d_gamai.mean(axis=0)
		grad['d_beta'+str(i)]=d_betai.mean(axis=0)
	return grad



#def img3d_convolution(X,kfilter):
#X=np.arange(2*4*4).reshape(2,4,4)
#kf=np.ones((3,3))
#y=img_convolution(X,kf)
#y2=maxpooling(y)
#print('X.shape=',X.shape)
#print('X=\n',X)
#print('y.shape=',y.shape)
#print('y=\n',y)
#print('y2.shape=',y2.shape)



n=1	
xin=mnist.test_img[n]
lab=mnist.test_lab[n].squeeze()
xin=xin.reshape(1,28,1,28)

print('params.keys()=\n',params.keys())
print('params[k0]=\n',params['k0'])
print('params[k1]=\n',params['k1'])
y=fp(xin,params,g)
print('y=\n',y.transpose(0,1,3,2))
print('y.shape=',y.shape)

#plt.imshow(xin,cmap='gray')
#plt.show()
#plt.imshow(y[0],cmap='gray')
#plt.show()


