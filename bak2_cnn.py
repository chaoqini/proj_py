##!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#import numba as np
#import numexpr as np
#from numba import jit
from timeit import timeit
import mnist
from matplotlib import pyplot as plt 
import pickle
import sys
import time
import copy


(imh,imw,lays,convk)=(28,28,4,3)
#(imh,imw,lays,convk)=(8,8,4,3)

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
	assert(X.ndim==4)
	exp=np.exp(X-np.max(X,axis=(1,2,3),keepdims=1))
	expsum=np.sum(exp,axis=(1,2,3),keepdims=1)
	return exp/expsum
def cross_entropy(X,LAB,params,g,isvalid=0):
	assert(X.ndim==4)
#	print('cross_entropy')
	Y=fp(X,params,g)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
	YL[np.arange(ba),LAB.reshape(ba),0,0]=1
	LOSS=-np.sum(YL*np.log(Y))
	cost=LOSS/ba
	if isvalid==0:
		return cost
	else:
#		print('Y.shape=',Y.shape)
#		print('Y[1]=',Y[1])
#		print('YL[1]=',YL[1])
#		y1d_max=np.max(Y,axis=1,keepdims=1)
#		Y=np.trunc(Y/y1d_max)
#		cmp=(Y==YL)
#		print('cmp.shape=',cmp.shape)

#		correct=np.trunc(np.sum(cmp,axis=1)/cmp.shape[1])
#		print('correct.shape=',correct.shape)
		correct=(np.argmax(Y,1)==np.argmax(YL,1))
		valid_per=correct.sum()/len(correct)
		return (cost,valid_per,correct)
## ==========
#@jit(nopython=True)
def im2col(im,k=3,byte=1):
	assert(im.ndim==4 and im.shape[-1]==1)
	p=int(k/2)
	(ba,h,w,non)=im.shape
	imp=np.pad(im,((0,0),(p,p),(p,p),(0,0)))
	(ba,hp,wp,n)=imp.shape
	strd=(hp*wp,wp,1,wp,1)
	strd=(i*byte for i in strd)
	col=np.lib.stride_tricks.as_strided(imp,shape=(ba,h,w,k,k),strides=strd)
	col=col.reshape(ba,h,w,k*k)
	return col

def im2col2(im,k=3):
	assert(im.ndim==4 and im.shape[-1]==1)
	p=int(k/2)
	imp=np.pad(im,((0,0),(p,p),(p,p),(0,0)))
	(ba,h,w,n)=imp.shape
	(col_r,col_c)=(h-k+1,w-k+1)
	col=np.zeros((ba,col_r,col_c,k*k))
#	jit(nopython=1)
	for r in range(col_r):
		for c in range(col_c):
			col[:,r,c,:]=imp[:,r:r+k,c:c+k,0].reshape(imp.shape[0],-1)
	return col
def conv(A,kf):
	assert(A.ndim==4 and A.shape[-1]==1 and kf.ndim==2)
	C=im2col(A,kf.shape[0])
	Z=C@kf.reshape(-1,1)
	return Z
#def col2im(col,k=3,p=-1):
def col2im(col,p=-1):
	assert(col.ndim==4)
	(ba,h,w,kk)=col.shape
	k=int(kk**0.5)
	if p==-1: p=int(k/2)
	im=np.zeros((ba,h,w,1))
	for r in range(h):
		for c in range(w):
			im[:,r,c,0]=col[:,r,c,int(k*(p+0.5))]
	return im


def maxpooling(X,k=2):
	if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
	assert(X.ndim==3)
	(h,w)=X.shape[1:3]
	(h,w)=(int(h/k),int(w/k))
	cols=np.zeros((X.shape[0],h,k*k,w))
	for r in range(h):
		for c in range(w):
			cols[:,r,:,c]=X[:,k*r:k*r+k,k*c:k*c+k].reshape(X.shape[0],-1)
	maxcols=np.max(cols,axis=2)
	return maxcols

def init_params(lays=lays,k=convk,nh=imh,nw=imw,ny=10,func=0,seed=0):
	np.random.seed(seed)
	if func==0:  (func,func_d)=(relu,relu_d)
	(params_init,g,g_d,l2_grad)=({},[],[],{})
	for i in range(lays):
		if i==lays-1:
			params_init['w'+str(i)]=np.random.randn(nh,nw,ny)*1e-3
		else:
			params_init['k'+str(i)]=np.random.randn(k,k)
		params_init['gama'+str(i)]=np.array(1.0).reshape(1,1)
		params_init['beta'+str(i)]=np.array(0.0).reshape(1,1)
		g.append(func)
		g_d.append(func_d)
	g[-1]=softmax;g.append(cross_entropy)
	params=copy.deepcopy(params_init)
	return (params,params_init,g,g_d)
(params,params_init,g,g_d)=init_params()

def fp(X,params,g,isop=0,e=1e-8):
	assert(X.ndim==4)
	ba=X.shape[0]
	(l,OP)=(int(len(params)/3),{})
	OP['A-1']=X
	for i in range(l) :
		Ai_1=OP['A'+str(i-1)]
		if i==l-1:
			wi=params['w'+str(i)]
			Zi=np.einsum('mhwp,hwo->mop',Ai_1,wi)
			Zi=Zi.reshape(Zi.shape+(1,))
		else:
			ki=params['k'+str(i)]
			coli_1=im2col(Ai_1,ki.shape[1])
			Zi=coli_1@(ki.reshape(-1,1))
			OP['C'+str(i-1)]=coli_1
		gamai=params['gama'+str(i)]
		betai=params['beta'+str(i)]
		ui=Zi.mean((1,2,3),keepdims=1)
		vi=Zi.var((1,2,3),keepdims=1)
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
	tb=time.time()
	tt={}
	assert(X.ndim==4)
	(Y,OP)=fp(X,params,g,isop=1)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
	YL[np.arange(ba),LAB.reshape(ba),0,0]=1
	(l,d_,grad)=(int(len(params)/3),{},{})
	tt['1']=(time.time()-tb)*1e3
	for i in range(l-1,-1,-1):
		tt[str(i)+'_0']=time.time()
		if i==l-1: wi=params['w'+str(i)]
		else: ki=params['k'+str(i)]
		gamai=params['gama'+str(i)]
		betai=params['beta'+str(i)]
		ui=OP['u'+str(i)]
		vi=OP['v'+str(i)]
		Xi=OP['X'+str(i)]
		tt[str(i)+'_1']=(time.time()-tb)*1e3
		if i==l-1: d_Yi=Y-YL
		else: d_Yi=d_['Y'+str(i)]
		tt[str(i)+'_2']=(time.time()-tb)*1e3
		d_Xi=gamai*d_Yi
		XX=np.einsum('mijn,mkln->mijkln',Xi,Xi)
		Imm=np.ones((XX.shape))
		mmE=np.zeros((XX.shape))
		np.einsum('mijijn->mijn',mmE)[:]=mmE.shape[1]*mmE.shape[2]
		tt[str(i)+'_3']=(time.time()-tb)*1e3
		vi=vi.reshape(vi.shape+(1,1))
		dXi_Zi=(mmE-Imm-XX)/(mmE.shape[1]*mmE.shape[2]*(vi+e)**0.5)
		d_Zi=np.einsum('mklijn,mijn->mkln',dXi_Zi,d_Xi)
		tt[str(i)+'_4']=(time.time()-tb)*1e3
		if i==l-1: 
			Ai_1=OP['A'+str(i-1)]
			d_Ai_1=np.einsum('mopq,hwo->mhwq',d_Zi,wi)
			d_wi=np.einsum('mhwp,mopq->mhwo',Ai_1,d_Zi)
			grad['d_w'+str(i)]=d_wi.mean(0)
		else:
			d_Ai_1=conv(d_Zi,np.flip(ki))
			Ci_1=OP['C'+str(i-1)]
			d_ki=np.einsum('mhwk,mhwn->mkn',Ci_1,d_Zi)
			d_ki=d_ki.reshape((d_ki.shape[0],)+ki.shape)
			grad['d_k'+str(i)]=d_ki.mean(0)
#			d_ki=np.einsum('mhwk,mhwn->k',Ci_1,d_Zi)
#			d_ki=d_ki.reshape(ki.shape)
#			grad['d_k'+str(i)]=d_ki*Ci_1.shape[0]
		tt[str(i)+'_5']=(time.time()-tb)*1e3
		if i>=1:
			Yi_1=OP['Y'+str(i-1)]
			d_Yi_1=g_d[i-1](Yi_1)*d_Ai_1
			d_['Y'+str(i-1)]=d_Yi_1
		tt[str(i)+'_6']=(time.time()-tb)*1e3
		d_gamai=(d_Yi*Xi).sum()/d_Yi.shape[0]
		d_betai=(d_Yi).sum()/d_Yi.shape[0]
		grad['d_gama'+str(i)]=d_gamai
		grad['d_beta'+str(i)]=d_betai
		tt[str(i)+'_7']=(time.time()-tb)*1e3
	print('tt=\n',tt)
	return grad
	
def slope(x,lab,params,g,dv=1e-5):
	slp={}
	pt=copy.deepcopy(params)
	for (k,v) in pt.items():
		slp['d_'+k]=[]
		nloop=0
		for i in np.nditer(v,op_flags=['readwrite']):
			nloop+=1
			vbak=i*1
			i[...]=vbak-dv
			l1=g[-1](x,lab,pt,g)
			i[...]=vbak+dv
			l2=g[-1](x,lab,pt,g)
			kk=(l2-l1)/(2*dv)
			slp['d_'+k].append(kk)
			i[...]=vbak
		slp['d_'+k]=np.array(slp['d_'+k]).reshape(v.shape)
	iseq=1
	for k in params.keys():
		iseq=iseq&(np.all(pt[k]==params[k])) 
	assert(iseq==1)
	return slp

## ==========
def grad_check(x,lab,params,g,g_d,dv=1e-5):
	y2=slope(x,lab,params,g,dv)
	y1=bp(x,lab,params,g,g_d)
	(abs_error,ratio_error)=({},{})
	for (k,v) in y1.items():
		v1=v
		v2=y2[k]
		l2_v1=np.linalg.norm(v1)
		l2_v2=np.linalg.norm(v2)
		l2_v1d2=np.linalg.norm(v1-v2)
		abs_error[k]=l2_v1d2
		ratio_error[k]=l2_v1d2/(l2_v1+l2_v2)
	for (k,v) in y1.items():
		if v.size<12:
			print('grad[%s]=\n'%k,y1[k])
			print('slope[%s]=\n'%k,y2[k])
	print('grad_check: abs_error=\n',abs_error)
	print('grad_check: ratio_error=\n',ratio_error)
	return (ratio_error,abs_error)

## ==========
def update_params(params,grad,lr=0.01):
	for k in params.keys():
		params[k]-=lr*grad['d_'+k]
	return params
def init_adam(params):
	(v,s)=({},{})
	for k in params.keys():
		v[k] = np.zeros(params[k].shape)
		s[k] = np.zeros(params[k].shape)
	return (v,s)
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
## ==========

def batch_train(params,g,g_d,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1):
	max_batches=int(len(mnist.train_img)/batch)
	if batches<1: batches=max_batches
	batches=min(max_batches,int(batches))
	X=mnist.train_img[:batch*batches]
	LAB=mnist.train_lab[:batch*batches]
	X=X.reshape((-1,batch)+X.shape[1:])
	LAB=LAB.reshape((-1,batch)+LAB.shape[1:])
	print('Train input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
	(cost,lra)=([],[])
	print('Training bath running ...')
	for i in range(len(X)):
		pn=i%(max(int(len(X)/10),1))
		if pn==0 or i==len(X)-1:
			print('Training iteration number = %s/%s'%(i,len(X)))
		Xi=np.expand_dims(X[i],-1)
		LABi=np.expand_dims(LAB[i],-1)
		grad=bp(Xi,LABi,params,g,g_d)
		lr=lr0*klr**i
		lra.append(lr)
		if i==0: (v,s)=init_adam(params)
		else: (params,v,s)=update_params_adam(params,grad,lr,v,s,i)
		(cost_i,valid_per,correct)=g[-1](Xi,LABi,params,g,1)
		cost.append(cost_i)
#		if isl2grad==1:
#			for (kt,vt) in grad.items():
#				l2=np.linalg.norm(vt)/vt.size
#				bnn.l2_grad[kt].append(l2)
#		if (pn==0 or i==len(X)-1) and istime!=0:
#			te=time.time()
#			tspd=(te-tb)*1000
#			print('the spending time of %s/%s batch is %s mS'%(i,len(X),tspd))
#		if isl2grad==1:
	cost=np.array(cost)
	lra=np.array(lra)
#	 cost=np.array(cost)[50:-1]
	if isplot!=0:
		plt.figure()
		plt.subplot(311)
		plt.plot(lra)
		plt.ylabel('lra')
		plt.subplot(312)
		plt.plot(cost)
		plt.ylabel('Cost')
		plt.xlabel('Iterations *%s'%batch)
		var_title=(lr0,klr,batch)
		title='lr0=%.3e\n klr=%s\n batch=%s\n'%var_title
		plt.title(title,loc='left')
#		i=0
#		for k in l2_grad.keys():
#			i=i+1
##			if 'w' in k or 'beta' in k:
#			if not(k[2]=='b' and k[3].isdigit()):
##				plt.subplot(len(l2_grad),1,i)
#				plt.figure()
#				plt.plot(bnn.l2_grad[k])
#				plt.ylabel(k)
#		plt.xlabel('Iterations *%s'%batch)
##		var_title=(lr0,klr,batch)
##		title='lr0=%.3e\n klr=%s\n batch=%s\n'%var_title
#		title='l2_grad'
#		plt.title(title,loc='left')
#		plt.show()
	return (params,lra[-1])

## ==========
def valid(params,g,batch=0,batches=0):
	if batch<1: batch=100
	max_batches=int(len(mnist.valid_img)/batch)
	if batches<1: batches=max_batches
	batches=min(max_batches,int(batches))
	X=mnist.valid_img[:batch*batches]
	LAB=mnist.valid_lab[:batch*batches]
	X=X.reshape((-1,batch)+X.shape[1:])
	print('X.shape=',X.shape)
#	 X=bnn.normalize(X)
	LAB=LAB.reshape((-1,batch)+LAB.shape[1:])
#	 print('Valid Input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
	(cost,valid_per,correct)=([],[],[])
#	 print('Valid batch running ...')
	for i in range(len(X)):
		pn=i%(max(int(len(X)/10),1))
		Xi=np.expand_dims(X[i],-1)
#		print('X%s.shape='%i,Xi.shape)
#		LABi=np.expand_dims(LAB[i],-1)
#def cross_entropy(X,LAB,params,g,isvalid=0):
		(cost_i,valid_per_i,correct_i)=g[-1](Xi,LAB[i],params,g,1)
#		print('cost_%s='%i,cost_i)
#		print('valid_per_%s='%i,valid_per_i)
#		print('correct_%s.shape='%i,correct_i.shape)
#		print('correct_%s='%i,correct_i.T)
		cost.append(cost_i);
		valid_per.append(valid_per_i);
		correct.append(correct_i);
	cost=np.array(cost)
	valid_per=np.array(valid_per)
	correct=np.array(correct)
	valid_per=valid_per.sum()/len(valid_per)
	print('Valid percent is : %.2f%%'%(valid_per*100))
	return (valid_per,correct) 
## ==========
## ==========
def valid_train(params,g,batch=0,batches=0):
	if batch<1: batch=100
	max_batches=int(len(mnist.train_img)/batch)
	if batches<1: batches=max_batches
	batches=min(max_batches,int(batches))
	X=mnist.train_img[:batch*batches]
	LAB=mnist.train_lab[:batch*batches]
	X=X.reshape((-1,batch)+X.shape[1:])
	LAB=LAB.reshape((-1,batch)+LAB.shape[1:])
#	X=mnist.train_img[:batch*batches]
#	LAB=mnist.train_lab[:batch*batches]
#	X=X.reshape((-1,batch)+X.shape[1:])
#	LAB=LAB.reshape((-1,batch)+LAB.shape[1:])
#	 print('Valid_train Input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
	(cost,valid_per,correct)=([],[],[])
#	 print('Valid_train batch running ...')
	for i in range(len(X)):
		pn=i%(max(int(len(X)/10),1))
		Xi=np.expand_dims(X[i],-1)
		(cost_i,valid_per_i,correct_i)=g[-1](Xi,LAB[i],params,g,1)
		cost.append(cost_i)
		valid_per.append(valid_per_i)
		correct.append(correct_i)
	cost=np.array(cost)
	valid_per=np.array(valid_per)
	correct=np.array(correct)
	valid_per=valid_per.sum()/len(valid_per)
#	 forprint('Valid_train L2_normalize_%s = %s'%(k,L2))
	print('Valid_train percent is : %.2f%%'%(valid_per*100))
	return (valid_per,correct) 
## ==========
## ==========
def show(params,g,n=-1):
	if n==-1: n=np.random.randint(mnist.test_num)
	x=mnist.test_img[n]
	lab=mnist.test_lab[n]
	x=np.expand_dims(x,(0,-1))
#	lab=mnist.test_lab[n].squeeze()
	y=fp(x,params,g)
	y=np.argmax(y)
	print('Real lab number is :\t%s'%lab)
	print('Precdict number is :\t%s'%y)
	img=x.reshape(28,28)
	plt.imshow(img,cmap='gray')
	plt.show()
## ==========

def train_and_valid(params,g,g_d,lr0=2e-3,klr=0.9995,batch=20,batches=0,isplot=0,istime=0,ischeck=0,isl2grad=0):
	(params,lrend)=batch_train(params,g,g_d,lr0,klr,batch,batches,isplot,istime,isl2grad=isl2grad)
	(valid_per,correct)=valid(params,g)
	(valid_per2,correct2)=valid_train(params,g)
	if ischeck==1:
		print('Grade check running ...')
		x=mnist.test_img[0]
		lab=mnist.test_lab[0]
		grad_check(x,lab,params,g,g_d)
		print('Grade check end.')
	return (lrend,valid_per,valid_per2)

def hyperparams_test(params,params_init,g,g_d,nloop=8,lr0=2e-3,klr=0.9995,batch=40,batches=0,isupdate=0,isl2grad=1):
	print('heyperparams_test: ...')
	print('heyperparams_test: layers =',int(len(params)/3))
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
		(lrendi,v1,v2)=train_and_valid(params,g,g_d,lri,klr,batch,batches,isplot=1,isl2grad=isl2grad)
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
	plt.legend(loc='best')
	plt.show()
	return params

#def batch_train(params,g,g_d,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1):
#(params,lre)=batch_train(params,g,g_d,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1)
#show(params,g)
