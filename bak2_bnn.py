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


## ==========
class bnn:
	def init_params(lays=3,nnode=100,nx=28*28,ny=10,func=0,seed=0):
#		 print('init_params:')
		np.random.seed(seed)
		if func==0:  (func,func_d)=(relu,relu_d)
		(nl,params_init,g,g_d)=([],{},[],[])
		nl.append(nx)
		for i in range(lays-1): nl.append(nnode)
		nl.append(ny)
		for i in range(lays):
			params_init['b'+str(i)]=np.ones((nl[i+1],1))*0
			params_init['w'+str(i)]=np.random.randn(nl[i+1],nl[i])*1e-3
			params_init['beta'+str(i)]=np.ones((nl[i+1],1))*0
			params_init['gama'+str(i)]=np.random.randn(nl[i+1],1)*1e-3
			g.append(func)
			g_d.append(func_d)
		g[-1]=softmax;g.append(cross_entropy)
		params=copy.deepcopy(params_init)
		return (params,params_init,g,g_d)
	(params,params_init,g,g_d)=init_params()


## ==========
	def fp(X,params,g,isop=0,e=1e-8):
		if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
		assert(X.ndim==3)
		(l,OP)=(int(len(params)/4),{})
		OP['A-1']=X
		for i in range(l) :
			wi=params['w'+str(i)]
			bi=params['b'+str(i)]
			gamai=params['gama'+str(i)]
			betai=params['beta'+str(i)]
			Ai_1=OP['A'+str(i-1)]
			Zi=wi@Ai_1+bi
			ui=np.mean(Zi,axis=1,keepdims=1)
			vi=np.var(Zi,axis=1,keepdims=1)
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
## ==========
	def bp(X,LAB,params,g,g_d,e=1e-8):
		if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
		if LAB.ndim==2: LAB=LAB.reshape(tuple([1])+LAB.shape)
		assert(X.ndim==3); assert(LAB.ndim==3)
		(Y,OP)=bnn.fp(X,params,g,1)
		meye=np.array([np.eye(Y.shape[1])]*len(LAB))
		lab=LAB.reshape(-1)
		assert(Y.ndim==3);assert(lab.ndim==1);assert(meye.ndim==3)
		nbatch=np.arange(len(meye))
		YL=meye[nbatch,lab,:]
		YL=YL.reshape(YL.shape+tuple([1]))
		(l,d_,grad)=(int(len(params)/4),{},{})
		for i in range(l-1,-1,-1):
			wi=params['w'+str(i)]
			bi=params['b'+str(i)]
			gamai=params['gama'+str(i)]
			betai=params['beta'+str(i)]
			ui=OP['u'+str(i)]
			vi=OP['v'+str(i)]
			Xi=OP['X'+str(i)]
			if i>0: Yin1=OP['Y'+str(i-1)]; Ain1=OP['A'+str(i-1)]
			if i==l-1: d_Yi=Y-YL
			else: d_Yi=d_['Y'+str(i)]
			d_Xi=gamai*d_Yi
			mi=Xi.shape[1]
			dXi_Zi=(mi*np.eye(mi)-np.ones((mi,mi))-Xi@Xi.transpose(0,2,1))/(mi*(vi+e)**0.5)
			d_Zi=dXi_Zi.transpose(0,2,1)@d_Xi
			if i>0:
				d_Ain1=wi.T@d_Zi
				d_Yin1=g_d[i](Yin1)*d_Ain1
				d_['Y'+str(i-1)]=d_Yin1
				d_wi=d_Zi@Ain1.transpose(0,2,1)
			d_bi=d_Zi
			d_gamai=d_Yi*Xi
			d_betai=d_Yi
			grad['d_w'+str(i)]=d_wi.mean(axis=0)
			grad['d_b'+str(i)]=d_bi.mean(axis=0)
			grad['d_gama'+str(i)]=d_gamai.mean(axis=0)
			grad['d_beta'+str(i)]=d_betai.mean(axis=0)
		return grad
## ==========
	def slope(x,lab,params,g,dv=1e-5):
#		 print('slope:')
		slp={}
		pt=copy.deepcopy(params)
		for (k,v) in pt.items():
			slp['d_'+k]=np.zeros(v.shape)
			for i in range(len(v)):
				for j in range(len(v[i])):
					vb=v[i,j]
					v[i,j]=vb-dv
					l1=g[-1](x,lab,pt,g)
					v[i,j]=vb+dv
					l2=g[-1](x,lab,pt,g)
					v[i,j]=vb
					kk=(l2-l1)/(2*dv)
					slp['d_'+k][i,j]=kk
		iseq=1
		for k in params.keys():
#			 iseq=iseq&(np.any(pt[k]==params[k])) 
			iseq=iseq&(np.all(pt[k]==params[k])) 
		assert(iseq==1)
		return slp
## ==========
	def grad_check(x,lab,params,g,g_d,dv=1e-5):
#		 print('grad_check:')
		y1=bnn.bp(x,lab,params,g,g_d)
		y2=bnn.slope(x,lab,params,g,dv)
		abs_error={};ratio_error={}
		for (k,v) in y1.items():
			v1=v
			v2=y2[k]
			l2_v1=np.linalg.norm(v1)
			l2_v2=np.linalg.norm(v2)
			l2_v1d2=np.linalg.norm(v1-v2)
			abs_error[k]=l2_v1d2
			ratio_error[k]=l2_v1d2/(l2_v1+l2_v2)
#			 print('grad_check: %s abs_err= %s'%(k,abs_error[k]))
#			 print('grad_check: %s ratio_err= %s'%(k,ratio_error[k]))
		print('grad_check: abs_error=',abs_error)
		print('grad_check: ratio_error=',ratio_error)
		return (ratio_error,abs_error)
## ==========
	def update_params(params,grad,lr=0.1):
		for k in params.keys():
			params[k]-=lr*grad['d_'+k]
		return params
	def update_params_adam(params,grad,lr,v,s,t=1):
		(beta1,beta2,e)=(0.9,0.999,1e-8)
		for k in params.keys():
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
#		 AX=mnist.train_img
		e=1e-8 
		AX=X
		mean=AX.mean()
		std=AX.std()
		var=AX.var()
#		 return (X-mean)/(std+e)
		return (X-mean)/(var+e)**0.5
## ==========
#mnist.train_num=50000
def batch_train(params,g,g_d,lr0=2e-3,klr=0.9995,batch=40,batches=0,isplot=0,istime=0):
#	 if batch<1: batch=bnn.batch
	max_batches=int(len(mnist.train_img)/batch)
	if batches<1: batches=max_batches
	batches=min(max_batches,int(batches))
	X=mnist.train_img[:batch*batches]
	LAB=mnist.train_lab[:batch*batches]
	X=X.reshape((-1,batch)+X.shape[1:3])
	X=bnn.normalize(X)
	LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
	print('Train input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
	(cost,lra)=([],[])
	print('Training bath running ...')
	for i in range(len(X)):
		pn=i%(max(int(len(X)/10),1))
		if pn==0 or i==len(X)-1:
			print('Training iteration number = %s/%s'%(i,len(X)))
			if istime!=0:tb=time.time()
		grad=bnn.bp(X[i],LAB[i],params,g,g_d)
#		 lr=lr0/(1+i/100)
		lr=lr0*klr**i
		lra.append(lr)
		if i==0: (v,s)=bnn.init_adam(params)
		else: (params,v,s)=bnn.update_params_adam(params,grad,lr,v,s,i)
		(cost_i,valid_per,correct)=g[-1](X[i],LAB[i],params,g,1)
		cost.append(cost_i)
		if (pn==0 or i==len(X)-1) and istime!=0:
			te=time.time()
			tspd=(te-tb)*1000
			print('the spending time of %s/%s batch is %s mS'%(i,len(X),tspd))
#		 print('batch: cost_i=',cost_i)
	cost=np.array(cost)
	lra=np.array(lra)
#	 cost=np.array(cost)[50:-1]
	if isplot!=0:
		plt.figure()
		plt.subplot(211)
		plt.plot(lra)
		plt.ylabel('lra')
		plt.subplot(212)
		plt.plot(cost)
		plt.ylabel('Cost')
		plt.xlabel('Iterations *%s'%batch)
		var_title=(lr0,klr,batch)
		title='lr0=%.3e\n klr=%s\n batch=%s\n'%var_title
		plt.title(title,loc='left')
		plt.show()
	return (params,lra[-1])

## ==========
def valid(params,g,batch=0,batches=0):
	if batch<1: batch=100
	max_batches=int(len(mnist.valid_img)/batch)
	if batches<1: batches=max_batches
	batches=min(max_batches,int(batches))
	X=mnist.valid_img[:batch*batches]
	LAB=mnist.valid_lab[:batch*batches]
	X=X.reshape((-1,batch)+X.shape[1:3])
#	 X=bnn.normalize(X)
	LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
#	 print('Valid Input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
	(cost,valid_per,correct)=([],[],[])
#	 print('Valid batch running ...')
	for i in range(len(X)):
		pn=i%(max(int(len(X)/10),1))
#		 if pn==0 or i==len(X)-1:
#			 print('Valid iteration number = %s/%s'%(i,len(X)))
		(cost_i,valid_per_i,correct_i)=g[-1](X[i],LAB[i],params,g,1)
		cost.append(cost_i)
		valid_per.append(valid_per_i)
		correct.append(correct_i)
	cost=np.array(cost)
	valid_per=np.array(valid_per)
	correct=np.array(correct)
#	 print('Valid L2 norm:')
	valid_per=valid_per.sum()/len(valid_per)
#	 for (k,v) in params.items():
#		 L2=np.linalg.norm(v)/v.size
#		 print('Valid L2_normalize_%s = %s'%(k,L2))
	print('Valid percent is : %.2f%%'%(valid_per*100))
	return (valid_per,correct) 
## ==========
def valid_train(params,g,batch=0,batches=0):
	if batch<1: batch=100
	max_batches=int(len(mnist.train_img)/batch)
	if batches<1: batches=max_batches
	batches=min(max_batches,int(batches))
	X=mnist.train_img[:batch*batches]
	LAB=mnist.train_lab[:batch*batches]
	X=X.reshape((-1,batch)+X.shape[1:3])
#	 X=bnn.normalize(X)
	LAB=LAB.reshape((-1,batch)+LAB.shape[1:3])
#	 print('Valid_train Input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
	(cost,valid_per,correct)=([],[],[])
#	 print('Valid_train batch running ...')
	for i in range(len(X)):
		pn=i%(max(int(len(X)/10),1))
#		 if pn==0 or i==len(X)-1:
#			 print('Valid_train iteration number = %s/%s'%(i,len(X)))
		(cost_i,valid_per_i,correct_i)=g[-1](X[i],LAB[i],params,g,1)
		cost.append(cost_i)
		valid_per.append(valid_per_i)
		correct.append(correct_i)
	cost=np.array(cost)
	valid_per=np.array(valid_per)
	correct=np.array(correct)
	valid_per=valid_per.sum()/len(valid_per)
#	 for (k,v) in params.items():
#		 L2=np.linalg.norm(v)/v.size
#		 print('Valid_train L2_normalize_%s = %s'%(k,L2))
	print('Valid_train percent is : %.2f%%'%(valid_per*100))
	return (valid_per,correct) 
## ==========
## ==========
def show(params,g,n=-1):
	if n==-1: n=np.random.randint(mnist.test_num)
	x=mnist.test_img[n]
	lab=mnist.test_lab[n].squeeze()
	y=bnn.fp(x,params,g)
	y=np.argmax(y)
	print('Real lab number is :\t%s'%lab)
	print('Precdict number is :\t%s'%y)
	img=x.reshape(28,28)
#	 plt.imshow(img)
	plt.imshow(img,cmap='gray')
	plt.show()
## ==========
## ==========

def train_and_valid(params,g,g_d,lr0=2e-3,klr=0.9995,batch=20,batches=0,isplot=0,istime=0,ischeck=0):
	(params,lrend)=batch_train(params,g,g_d,lr0,klr,batch,batches,isplot,istime)
	(valid_per,correct)=valid(params,g)
	(valid_per2,correct2)=valid_train(params,g)
	if ischeck==1:
		print('Grade check running ...')
		bnn.grad_check(x,params,lab)
		print('Grade check end.')
	return (lrend,valid_per,valid_per2)

def hyperparams_test(params,params_init,g,g_d,nloop=8,lr0=2e-3,klr=0.9995,batch=40,batches=0,isupdate=0):
	print('heyperparams_test: ...')
	print('heyperparams_test: layers =',int(len(params)/2))
	print('heyperparams_test: learning rate lr0 =',lr0)
	print('heyperparams_test: learning rate klr =',klr)
	print('heyperparams_test: batch =',batch)
	print('heyperparams_test: isupdate =',isupdate)
	for k in params.keys():
		print( 'params %s.shape='%k,params[k].shape)
	for i in range(len(g)):
		print('active function g[%d] is:'%i,g[i].__name__)
	(v1a,v2a)=([],[])
	lrendi=lr0
	for i in range(nloop):
		print('='*60)
		print('hyperparams_test runing iteration = %s/%s'%(i+1,nloop))
		if isupdate==0: 
			params=copy.deepcopy(params_init)
			lri=lr0*(1-i/nloop)
		else:
			lri=lrendi
		print('hyperparams_test runing: lri = %.3e'%lri)
		for (k,v) in params.items():
			L2=np.linalg.norm(v)/v.size
			print('Hyperparams_test: L2_normalize_%s = %.2e'%(k,L2))
		(lrendi,v1,v2)=train_and_valid(params,g,g_d,lri,klr,batch,batches,isplot=1)
		v1a.append(v1)
		v2a.append(v2)
#		 lr=lr0*klr**i
#		 if i==nloop-1: lrend=lrendi
	v1a=np.array(v1a)
	v2a=np.array(v2a)
	plt.figure()
	plt.subplot(211)
	plt.plot(v1a)
	plt.ylabel('v1a')
	plt.subplot(212)
	plt.plot(v2a)
	plt.ylabel('v2a')
	plt.legend(loc='best')
	plt.show()

(params,params_init,g,g_d)=bnn.init_params(lays=2,nnode=100)
hyperparams_test(params,params_init,g,g_d,nloop=9,lr0=2e-3,klr=0.9995,batch=40,isupdate=1)
#hyperparams_test(params,params_init,g,g_d,nloop=6,lr0=2e-3,klr=0.9995,batch=30)




