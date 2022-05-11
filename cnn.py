##!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
from matplotlib import pyplot as plt 
import pickle
import sys
import time
import copy


(hhh,www)=(6,6)

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
	Y=fp(X,params,g)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
	YL[np.arange(ba),LAB.reshape(ba),0,0]=1
#	print('Y=',Y.transpose(0,3,2,1))
#	print('YL=',YL.transpose(0,3,2,1))
#	meye=np.array([np.eye(Y.shape[1])]*len(LAB))
#	lab=LAB.reshape(-1)
#	nbatch=np.arange(len(meye))
#	YL=meye[nbatch,lab,:]
#	YL=YL.reshape(YL.shape+tuple([1]))
#	LOSS=-np.sum(YL*np.log(Y),axis=(1,2,3))
	LOSS=-np.sum(YL*np.log(Y))
	cost=LOSS/ba
#	print('cost=',cost)
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
	if X.ndim==2: X=X.reshape((1,)+X.shape+(1,))
	elif X.ndim==3: X=X.reshape(X.shape+(1,))
	assert(X.ndim==4)
	kf=kfilter.shape[1]
	p=int(kf/2)
	k1=kfilter.reshape(-1,1)
	X=np.pad(X,((0,0),(p,p),(p,p),(0,0)))
	(ba,h,w,Non)=X.shape
	(colk_r,colk_c)=(h-kf+1,w-kf+1)
	colk=np.zeros((ba,colk_r,colk_c,kf*kf))
	for r in range(colk_r):
		for c in range(colk_c):
			colk[:,r,c,:]=X[:,r:r+kf,c:c+kf,0].reshape(X.shape[0],-1)
	col1=colk@k1
	return col1

def im2col(im,k):
	assert(im.ndim==4 and im.shape[-1]==1)
	p=int(k/2)
	imp=np.pad(im,((0,0),(p,p),(p,p),(0,0)))
	(ba,h,w,n)=imp.shape
	(col_r,col_c)=(h-k+1,w-k+1)
	col=np.zeros((ba,col_r,col_c,k*k))
	for r in range(col_r):
		for c in range(col_c):
			col[:,r,c,:]=imp[:,r:r+k,c:c+k,0].reshape(imp.shape[0],-1)
	return col
	
def col2im(col,k,p=-1):
	assert(col.ndim==4)
	if p==-1: p=int(k/2)
	(ba,h,w,kk)=col.shape
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
#	print('h=',h)
#	print('w=',w)
	for r in range(h):
		for c in range(w):
			cols[:,r,:,c]=X[:,k*r:k*r+k,k*c:k*c+k].reshape(X.shape[0],-1)
	maxcols=np.max(cols,axis=2)
	return maxcols

def init_params(lays=3,kr=3,nk=2,nh=hhh,nw=www,ny=10,func=0,seed=0):
#def init_params(lays=3,kr=3,nk=2,nh=28,nw=28,ny=10,func=0,seed=0):
#		 print('init_params:')
	np.random.seed(seed)
	if func==0:  (func,func_d)=(relu,relu_d)
	(params_init,g,g_d,l2_grad)=({},[],[],{})
	for i in range(lays):
		if i==lays-1:
			params_init['w'+str(i)]=np.random.randn(nh,nw,ny)*1e-3
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
#	if X.ndim==2: X=X.reshape((1,)+X.shape+(1,))
#	elif X.ndim==3: X=X.reshape(X.shape+(1,))
	assert(X.ndim==4)
	(l,OP)=(int(len(params)/3),{})
	OP['A-1']=X
	for i in range(l) :
		Ai_1=OP['A'+str(i-1)]
		if i==l-1:
			wi=params['w'+str(i)]
#			Ai_1=Ai_1.reshape(Ai_1.shape[0],-1,1,1)
#			Zi=wi@Ai_1
			Zi=np.einsum('mhwp,hwo->mop',Ai_1,wi)
			Zi=Zi.reshape(Zi.shape+(1,))
#			print('Ai_1.shape=',Ai_1.shape)
#			print('Zi.shape=',Zi.shape)
		else:
			ki=params['k'+str(i)]
			coli_1=im2col(Ai_1,ki.shape[1])
#			Zi=img_convolution(Ai_1,ki)
			Zi=coli_1@(ki.reshape(-1,1))
			OP['C'+str(i-1)]=coli_1
		gamai=params['gama'+str(i)]
		betai=params['beta'+str(i)]
#		print('1: i=',i)
		ui=np.mean(Zi,axis=(1,2,3),keepdims=1)
		vi=np.var(Zi,axis=(1,2,3),keepdims=1)
		Xi=(Zi-ui)/(vi+e)**0.5
		Yi=gamai*Xi+betai
#		print('Yi.shape=',Yi.shape)
		Ai=g[i](Yi)
#		print('Ai.shape=',Ai.shape)
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
#	if X.ndim==2: X=X.reshape((1,)+(X.shape[0],)+(1,)+(X.shape[-1],))
#	elif X.ndim==3 and X.shape[-2]!=1: X=X.reshape(X.shape[:-1]+(1,)+X.shape[-1:])
#	assert(X.ndim==4)
#	if X.ndim==2: X=X.reshape(tuple([1])+X.shape)
#	if LAB.ndim==2: LAB=LAB.reshape((1,)+LAB.shape)
#	assert(X.ndim==4 and LAB.ndim==4)
	assert(X.ndim==4)
	(Y,OP)=fp(X,params,g,isop=1)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
#	YL[np.arange(ba),0,LAB.reshape(ba),0]=1
#	YL[np.arange(ba),LAB.reshape(ba),0,0]=1
	YL[np.arange(ba),LAB.reshape(ba),0,0]=1
	(l,d_,grad)=(int(len(params)/3),{},{})
	print('l=',l)
	for i in range(l-1,-1,-1):
		if i==l-1: wi=params['w'+str(i)]
		else: ki=params['k'+str(i)]
		gamai=params['gama'+str(i)]
		betai=params['beta'+str(i)]
		ui=OP['u'+str(i)]
		vi=OP['v'+str(i)]
		Xi=OP['X'+str(i)]
		if i==l-1: d_Yi=Y-YL
		else: d_Yi=d_['Y'+str(i)]
#		print('loop 1: i=',i)
		d_Xi=gamai*d_Yi
#		(batch,mx,nx)=Xi.shape
		print('Xi.shape bf =',Xi.shape)
#		Xi=Xi.reshape(Xi.shape[0],-1,1,1)
#		print('Xi.shape=',Xi.shape)
		XX=np.einsum('mijn,mkln->mijkln',Xi,Xi)
#		print('XX.shape=',XX.shape)
		Imm=np.ones((XX.shape))
		mmE=np.zeros((XX.shape))
		np.einsum('mijijn->mijn',mmE)[:]=mmE.shape[1]*mmE.shape[2]
#		print('bf vi.shape=',vi.shape)
		vi=vi.reshape(vi.shape+(1,1))
#		print('vi.shape=',vi.shape)
		dXi_Zi=(mmE-Imm-XX)/(mmE.shape[1]*mmE.shape[2]*(vi+e)**0.5)
#		d_Zi=dXi_Zi.transpose(0,2,1,3)@d_Xi
#		print('dXi_Zi.shape=',dXi_Zi.shape)
#		print('d_Xi.shape=',d_Xi.shape)
#		d_Zi=np.einsum('mijkln,mkln->mijn',dXi_Zi,d_Xi)
#		d_Zi=np.einsum('mijkln,mijn->mkln',dXi_Zi,d_Xi)
		d_Zi=np.einsum('mklijn,mijn->mkln',dXi_Zi,d_Xi)
#		print('d_Zi.shape=',d_Zi.shape)
#		print('l=',l)
		if i==l-1: 
#			d_Ain1=wi.T@d_Zi
			Ai_1=OP['A'+str(i-1)]
#			print('Ai_1.shape=',Ai_1.shape)
#			d_Ai_1=np.einsum('ij,mikn->mjkn',wi,d_Zi)
			d_Ai_1=np.einsum('mopq,hwo->mhwq',d_Zi,wi)
#			d_Ai_1=d_Ai_1.reshape(Ai_1.shape)
#			d_wi=d_Zi@Ain1.transpose(0,2,1,3)
#			d_Zi@Ain1.transpose(0,2,1,3)
#			Ai_1_re=Ai_1.reshape(Ai_1.shape[0],-1,1,1)
#			print('i=',i)
#			print('Ai_1.size=',Ai_1.size)
#			print('d_Zi.shape=',d_Zi.shape)
#			print('Ai_1_re.shape=',Ai_1_re.shape)
			d_wi=np.einsum('mhwp,mopq->mhwo',Ai_1,d_Zi)
#			print('d_wi.shape=',d_wi.shape)
			grad['d_w'+str(i)]=d_wi.mean(axis=0)
		else:
#			d_Zi=np.expand_dims(d_Zi,axis=-2)
			kir=ki.reshape(-1,1)
			d_Ci_1=d_Zi@kir.T
#			print('d_Ci_1.shape=',d_Ci_1.shape)
#			print('ki.shape=',ki.shape)
			d_Ai_1=col2im(d_Ci_1,ki.shape[1])
			Ci_1=OP['C'+str(i-1)]
			d_ki=np.einsum('mhwk,mhwn->mkn',Ci_1,d_Zi)
			d_ki=d_ki.reshape(d_ki.shape[0],ki.shape[1],-1)
			grad['d_k'+str(i)]=d_ki.mean(axis=0)
		if i>=1:
#			print('3: i=',i)
			Yi_1=OP['Y'+str(i-1)]
#			print('Yi_1.shape=',Yi_1.shape)
#			print('g_d[i-1](Yi_1).shape=',g_d[i-1](Yi_1).shape)
#			print('d_Ai_1.shape=',d_Ai_1.shape)
			d_Yi_1=g_d[i-1](Yi_1)*d_Ai_1
			d_['Y'+str(i-1)]=d_Yi_1
		d_gamai=(d_Yi*Xi).sum()
		d_betai=(d_Yi).sum()
		grad['d_gama'+str(i)]=d_gamai
		grad['d_beta'+str(i)]=d_betai
	return grad

	
def slope(x,lab,params,g,dv=1e-5):
	slp={}
	pt=copy.deepcopy(params)
	for (k,v) in pt.items():
		print('key of pt is:',k)
		slp['d_'+k]=[]
		for i in np.nditer(v,op_flags=['readwrite']):
			vbak=i*1
			i[...]=vbak-dv
#			print('i1=',i)
			l1=g[-1](x,lab,pt,g)
#			print('l1=',l1)
			i[...]=vbak+dv
#			print('i2=',i)
			l2=g[-1](x,lab,pt,g)
#			print('l2=',l2)
			kk=(l2-l1)/(2*dv)
#			print('kk=',kk)
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
	y1=bp(x,lab,params,g,g_d)
	y2=slope(x,lab,params,g,dv)
	abs_error={};ratio_error={}
	for (k,v) in y1.items():
		v1=v
		v2=y2[k]
		l2_v1=np.linalg.norm(v1)
		l2_v2=np.linalg.norm(v2)
		l2_v1d2=np.linalg.norm(v1-v2)
		abs_error[k]=l2_v1d2
		ratio_error[k]=l2_v1d2/(l2_v1+l2_v2)
	print('grad_check: abs_error=',abs_error)
	print('grad_check: ratio_error=',ratio_error)
	return (ratio_error,abs_error)

##(ba,h,w,ki)=(2,8,7,5)
#(ba,h,w,ki)=(2,28,28,3)
#aa=np.arange(ba*h*w).reshape(ba,h,w,1)+1
#cc=im2col(aa,ki)
#bb=col2im(cc,ki)
#print('aa=\n',aa.transpose(0,3,1,2))
#print('bb=\n',bb.transpose(0,3,1,2))
#print('cc=\n',cc)
#print('aa.shape=',aa.shape)
#print('bb.shape=',bb.shape)
#print('cc.shape=',cc.shape)
#


n=1	
#xin=mnist.test_img[n]
#lab=mnist.test_lab[n].squeeze()
#xin=xin.reshape(1,28,28,1)
xin=mnist.test_img[n:n+2]
lab=mnist.test_lab[n:n+2].squeeze()
#xin=xin.reshape(2,28,28,1)
np.random.seed(1)
xin=np.random.randn(2,hhh,www,1)
lab=lab.reshape(2,1,1)

print('params.keys()=\n',params.keys())
print('params[k0]=\n',params['k0'])
print('params[k1]=\n',params['k1'])
#y=fp(xin,params,g)
(y,OP)=fp(xin,params,g,isop=1)
#def bp(X,LAB,params,g,g_d,e=1e-8):
grad=bp(xin,lab,params,g,g_d)
#def slope(x,lab,params,g,dv=1e-5):
slp=slope(xin,lab,params,g)

#print('y=\n',y.transpose(0,1,3,2))
print('y=\n',y.transpose(0,3,2,1))
print('y.shape=',y.shape)
print('grad.keys()=\n',grad.keys())
print('slp.keys()=\n',slp.keys())
print('grad[d_w2].shape=\n',grad['d_w2'].shape)
#print('slope[d_w2].shape=\n',slope['d_w2'].shape)
print('grad[d_w2]=\n',grad['d_w2'])
#print('slope[d_w2]=\n',slope['d_w2'])
#print('grad[d_k1]=\n',grad['d_k1'])
#print('slope[d_k1]=\n',slope['d_k1'])
#print('slope[d_gama1]=\n',slope['d_gama1'])
#for k,v in slope.items():
#	print('slope[%s]=\n'%k,v)


grad_check(xin,lab,params,g,g_d)

#def grad_check(x,lab,params,g,g_d,dv=1e-5):
#plt.imshow(xin,cmap='gray')
#plt.show()
#plt.imshow(y[0],cmap='gray')
#plt.show()


