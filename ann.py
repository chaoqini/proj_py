#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#import numba as np
#import numexpr as np
#from numba import jit
#from timeit import timeit
import mnist
from matplotlib import pyplot as plt 
import pickle
import sys
import time
import copy

#(lays,imba,imch,imh,imw,convk,km,minhw)=(5,2,1,28,28,3,2,4)
#(lays,imba,imch,imh,imw,convk,km,minhw)=(5,2,2,28,28,3,2,4)
#(lays,imba,imchin,dimch,imh,imw,convk,km,minhw)=(5,2,1,2,28,28,3,2,28)
(lays,imba,imchin,dimch,imh,imw,convk,km,minhw)=(4,2,2,1,28,28,3,2,28)
#(lays,imba,imchin,dimch,imh,imw,convk,km,minhw)=(5,1,1,1,6,6,3,2,8)
#imchs=[1,]
#(lays,imba,imchin,dimch,imh,imw,convk,km,minhw)=(5,2,2,2,8,8,3,2,8)
np.random.seed(0)
## ==========
def Relu(Yi,d_Ai=None,Ai=None):
	if d_Ai is not None:
		if Ai is None: Ai=Yi.copy(); Ai[Ai<=0]=0
		dAi_dYi=Ai.copy()
		dAi_dYi[dAi_dYi>0]=1
		d_Yi=dAi_dYi*d_Ai
		return d_Yi
	else: 
		Ai=Yi.copy(); Ai[Ai<=0]=0
		return Ai
def softmax(x): 
	xmax=np.max(x,(-2,-1),keepdims=1)
	exp=np.exp(x-xmax)
	expsum=np.sum(exp,(-2,-1),keepdims=1)
	y=exp/expsum
	return y
def cross_entropy(x,lab,params,g,isvalid=0):
	y=fp(x,params,g)
	ba=y.shape[0]
	yl=np.zeros(y.shape)
	yl[np.arange(ba),0,lab.reshape(ba),0]=1
	loss=-np.sum(yl*np.log(y))
	cost=loss/ba
	if isvalid==0:
		return cost
	else:
		correct=(np.argmax(y,-2)==np.argmax(yl,-2))
		valid_per=correct.sum()/len(correct)
		return (cost,valid_per,correct)

#def init_params(lays=lays,k=convk,nh=imh,nw=imw,nk=2,ny=10,func=0,seed=0):
def init_params(lays=lays,k=convk,imchin=imchin,dimch=dimch,imh=imh,imw=imw,ch=-1,func=None):
	if func is None: func=Relu
	if ch==-1:
		ch=[1]
		for i in range(lays-1):ch.append(imchin+dimch*i)
		ch.append(10)
	(params_init,g,l2_grad)=({},[],{})
	print('init_params: ch=',ch)
	for i in range(lays):
		if i==lays-1:
				nh=int(np.log2(imh/minhw));nw=int(np.log2(imw/minhw))
				params_init['w'+str(i)]=np.random.randn(ch[i+1],ch[i],int(imh/km**nh),int(imw/km**nw))
		else:
			params_init['k'+str(i)]=np.random.randn(ch[i+1],ch[i],k,k)
			params_init['gama'+str(i)]=np.ones((ch[i+1],1,1))
			params_init['beta'+str(i)]=np.zeros((ch[i+1],1,1))
		g.append(func)
	g[-1]=softmax;g.append(cross_entropy)
	params=copy.deepcopy(params_init)
#	for k,v in params.items():
#		print('params: %s.shape='%k,v.shape)
	return (params,params_init,g)
(params,params_init,g)=init_params()

def im2col(im,k=3,s=1,pad=1):
		if k==1: col=np.expand_dims(im,(-2,-1))
		else:
			p=int(k/2)
			(ba,c,h,w)=im.shape
			if pad==1: imp=np.pad(im,((0,0),(0,0),(p,p),(p,p)))
			else: imp=im
			(ba,c,hp,wp)=imp.shape
			strd=(c*hp*wp,hp*wp,wp*s,s,wp,1)
			strd=(i*imp.itemsize for i in strd)
			col=np.lib.stride_tricks.as_strided(imp,shape=(ba,c,int((hp-k)/s)+1,int((wp-k)/s)+1,k,k),strides=strd)
		return col
def maxpooling(Z,k=2,d_M=None):
	e=1e-12
	Mkk=im2col(Z,k=2,s=2,pad=0)
	M=np.max(Mkk,(-2,-1))
	if d_M is not None:
		ZdM=(Mkk+1-Mkk.max((-2,-1),keepdims=1))
		ZdM[ZdM<1-e]=0
		(ba,c,hp,wp,khp,kwp)=ZdM.shape
		d_Z_Mkk=np.expand_dims(d_M,(-2,-1))*ZdM
		strd=(c*hp*khp*wp*kwp,hp*khp*wp*kwp,wp*kwp,1)
		strd=(i*d_Z_Mkk.itemsize for i in strd)
		d_Z=np.lib.stride_tricks.as_strided(d_Z_Mkk,shape=(ba,c,hp*khp,wp*kwp),strides=strd)
		return M,d_Z
	else: return M
## ==========
def fco(Ai_1,wi,d_Zi=None):
	if d_Zi is not None:
		d_Ai_1=np.einsum('bpoq,ochw->bchw',d_Zi,wi)
		d_wi=np.einsum('bchw,bpoq->bochw',Ai_1,d_Zi)
		return d_Ai_1,d_wi
	else: 
		Zi=np.einsum('bchw,ochw->bo',Ai_1,wi)
		Zi=np.expand_dims(Zi,(1,-1))
		return Zi
## ==========
def conv(Ai_1,ki,d_Zi=None,Ci_1=None):
	if d_Zi is not None: 
		ki_fl=np.flip(ki,(-2,-1))
		d_Zi_2col=im2col(d_Zi,ki_fl.shape[-1])
		d_Ai_1=np.einsum('bmhwij,mcij->bchw',d_Zi_2col,ki_fl)
		if Ci_1 is None: Ci_1=im2col(Ai_1,ki.shape[-1])
		d_ki=np.einsum('bchwij,bmhw->bmcij',Ci_1,d_Zi)
		return d_Ai_1,d_ki
	else:	
		Ci_1=im2col(Ai_1,ki.shape[-1])
		Zi=np.einsum('bchwij,mcij->bmhw',Ci_1,ki)
		return Zi,Ci_1
## ==========
def norm(Zi,gamai,betai,d_Yi=None,Xi=None):
	e=1e-8
	ui=Zi.mean((-2,-1),keepdims=1)
	vi=Zi.var((-2,-1),keepdims=1)
	if d_Yi is not None:
		d_Xi=gamai*d_Yi
		if Xi is None: Xi=(Zi-ui)/(vi+e)**0.5
		XX=np.einsum('bcij,bckl->bcijkl',Xi,Xi)
		Imm=np.ones((XX.shape))
		mmE=np.zeros((XX.shape))
		mm=XX.shape[-2]*XX.shape[-1]
		np.einsum('bcijij->bcij',mmE)[:]=mm
		vmm=np.expand_dims(vi,(-2,-1))
		dXi_dZi=(mmE-Imm-XX)/(mm*(vmm+e)**0.5)
		d_Zi=np.einsum('bcijkl,bckl->bcij',dXi_dZi,d_Xi)
		d_gamai=(d_Yi*Xi).sum((-2,-1),keepdims=1)
		d_betai=(d_Yi).sum((-2,-1),keepdims=1)
		return d_Zi,d_gamai,d_betai
	else: 
		Xi=(Zi-ui)/(vi+e)**0.5
		Yi=gamai*Xi+betai
		return Yi,Xi
## ==========
def fp(X,params,g,opin=None,pname=None,isop=0,e=1e-8):
#	print('fp begin..')
	ba=X.shape[0]
	(l,OP)=(int(len(params)/3)+1,{})
	OP['A-1']=X
	for i in range(l) :
		Ai_1=OP['A'+str(i-1)]
		if i<l-1:
			ki=params['k'+str(i)]
			if pname=='A'+str(i-1): Ai_1=opin
			Zi,Ci_1=conv(Ai_1,ki)
			gamai=params['gama'+str(i)]
			betai=params['beta'+str(i)]
			Yi,Xi=norm(Zi,gamai,betai)
			OP['C'+str(i-1)]=Ci_1
			OP['X'+str(i)]=Xi
		else:
			wi=params['w'+str(i)]
			if pname=='A'+str(i-1): Ai_1=opin
			Zi=fco(Ai_1,wi)
			Yi=Zi
		if pname=='Y'+str(i): Yi=opin
		Ai=g[i](Yi)
		OP['Z'+str(i)]=Zi
		OP['Y'+str(i)]=Yi
		OP['A'+str(i)]=Ai
	Y=OP['A'+str(l-1)]
#	print('fp end..')
	if isop!=0:	return Y,OP
	else:	return Y

## ==========
def bp(X,LAB,params,g,e=1e-8,isop=0):
#	print('bp begin..')
	(Y,OP)=fp(X,params,g,isop=1)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
	YL[np.arange(ba),0,LAB.reshape(ba),0]=1
	(l,d_,grad)=(int(len(params)/3)+1,{},{})
	for i in range(l-1,-1,-1):
		Ai_1=OP['A'+str(i-1)]
		if i==l-1: 
			wi=params['w'+str(i)]
			d_Yi=Y-YL
			d_Zi=d_Yi
			d_Ai_1,d_wi=fco(Ai_1,wi,d_Zi=d_Zi)
			grad['d_w'+str(i)]=d_wi.mean(0)
		else:
			ki=params['k'+str(i)]
			gamai=params['gama'+str(i)]
			betai=params['beta'+str(i)]
			Zi=OP['Z'+str(i)]
			Xi=OP['X'+str(i)]
			d_Yi=d_['Y'+str(i)]
			d_Zi,d_gamai,d_betai=norm(Zi,gamai,betai,d_Yi=d_Yi,Xi=Xi)
			Ci_1=OP['C'+str(i-1)]
			d_Ai_1,d_ki=conv(Ai_1,ki,d_Zi=d_Zi,Ci_1=Ci_1)
			grad['d_gama'+str(i)]=d_gamai.mean(0)
			grad['d_beta'+str(i)]=d_betai.mean(0)
			grad['d_k'+str(i)]=d_ki.mean(0)
		if i>=1:
			d_['Y'+str(i)]=d_Yi
			d_['Z'+str(i)]=d_Zi
			d_['A'+str(i-1)]=d_Ai_1
			Yi_1=OP['Y'+str(i-1)]
			Ai_1=OP['A'+str(i-1)]
			d_Yi_1=Relu(Yi_1,d_Ai=d_Ai_1,Ai=Ai_1)
			d_['Y'+str(i-1)]=d_Yi_1
#	print('bp end..')
	if isop!=0:	return (grad,d_)
	else: return grad
	
def slope(x,lab,params,g,dv=1e-5):
	slp={}
	pt=copy.deepcopy(params)
	for (k,v) in pt.items():
		print('slope: running %s.shape='%k,v.shape)
		slp['d_'+k]=[]
		nloop=0
		for i in np.nditer(v,op_flags=['readwrite']):
			nloop+=1
			if nloop%100==0 : print('slope: %s running loop=%s/%s'%(k,nloop,v.size))
			vbak=i*1
			i[...]=vbak-dv
			l1=g[-1](x,lab,pt,g)
			i[...]=vbak+dv
			l2=g[-1](x,lab,pt,g)
			kk=(l2-l1)/(2*dv)
#			if (v.size<200 and nloop%10==0) or nloop%100==0 : 
			if nloop%(int(nloop/4)+1) ==0 : 
				print('slope: %s[%s/%s] slope = %s'%(k,nloop,v.size,kk))
			slp['d_'+k].append(kk)
			i[...]=vbak
		slp['d_'+k]=np.array(slp['d_'+k]).reshape(v.shape)
	iseq=1
	for k in params.keys():
		iseq=iseq&(np.all(pt[k]==params[k])) 
	assert(iseq==1)
	return slp

def slope2(x,lab,params,g,pname=None,dv=1e-5):
	slp={}
	y,op=fp(x,params,g,isop=1)
	ba=y.shape[0]
	yl=np.zeros(y.shape)
	yl[np.arange(ba),0,lab.reshape(ba),0]=1
#	print('slope2: yl=\n',yl.squeeze())
	v=op[pname]
	slp['d_'+pname]=[]
	ni=0
	for i in np.nditer(v,op_flags=['readwrite']):
		ni+=1
		vbak=i*1
		i[...]=vbak-dv
		y1=fp(x,params,g,opin=v,pname=pname)
		i[...]=vbak+dv
		y2=fp(x,params,g,opin=v,pname=pname)
		loss1=-np.sum(yl*np.log(y1))
		loss2=-np.sum(yl*np.log(y2))
		kk=(loss2-loss1)/(dv*2)
		slp['d_'+pname].append(kk)
		i[...]=vbak
	slp['d_'+pname]=np.array(slp['d_'+pname]).reshape(v.shape)
	return slp['d_'+pname]
## ==========
def grad_check(x,lab,params,g,dv=1e-5):
	y1=bp(x,lab,params,g)
	y2=slope(x,lab,params,g,dv)
	(abs_error,ratio_error)=({},{})
	for (k,v) in y1.items():
		print('grad: %s.shape='%k,v.shape)
		print('slope: %s.shape='%k,y2[k].shape)
#	for (k,v) in y2.items():
#		print('slope: %s.shape='%k,v.shape)
	for (k,v) in y1.items():
		v1=v
		v2=y2[k]
		l2_v1=np.linalg.norm(v1)
		l2_v2=np.linalg.norm(v2)
		l2_v1d2=np.linalg.norm(v1-v2)
		abs_error[k]=l2_v1d2
		ratio_error[k]=l2_v1d2/(l2_v1+l2_v2)
	for (k,v) in y1.items():
#		if v.size<200:
		if v.size<400:
			print('=== grad_check : ')
			print('grad[%s].shape='%k,y1[k].shape)
			print('slope[%s].shape='%k,y2[k].shape)
			print('grad[%s]=\n'%k,y1[k].squeeze())
			print('slope[%s]=\n'%k,y2[k].squeeze())
	print('grad_check: abs_error=\n',abs_error)
	print('grad_check: ratio_error=\n',ratio_error)
	return (y1,y2)
def grad_check2(x,lab,params,g,pname=None,dv=1e-5):
	check_slp=slope2(x,lab,params,g,pname=pname,dv=dv)
	y1,d_=bp(x,lab,params,g,isop=1)
	print('check: d_keys()=\n',d_.keys())
	print('check: op_param=',pname)
	print('check: slope=\n',check_slp.squeeze())
	print('check: grad=\n',d_[pname].squeeze())
	return check_slp,d_[pname]
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

def batch_train(params,g,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1):
	max_batches=int(len(mnist.train_img)/batch)
	if batches<1: batches=max_batches
	batches=min(max_batches,int(batches))
	X=mnist.train_img[:batch*batches]
	LAB=mnist.train_lab[:batch*batches]
	X=X.reshape((-1,batch)+X.shape[1:])
	LAB=LAB.reshape((-1,batch)+LAB.shape[1:])
	print('Train input X.shape=%s, LAB.shape=%s'%(X.shape,LAB.shape))
	(cost,valid_per,correct,lra)=([],[],[],[])
	print('Training bath running ...')
	for i in range(len(X)):
		pn=i%(max(int(len(X)/2),1))
#		pn=i%(max(int(len(X)/10),1))
		if pn==0 or i==len(X)-1:
			print('Training iteration number = %s/%s'%(i,len(X)))
#		Xi=np.expand_dims(X[i],-1)
#		LABi=np.expand_dims(LAB[i],-1)
		t0=time.time()
#		print('%s bp begin time is:'%i,tb)
		grad=bp(X[i],LAB[i],params,g)
		t1=time.time()
		print('%s bp detal time dt1= %0.2fs'%(i,t1-t0))
		lr=lr0*klr**i
		lra.append(lr)
		if i==0: (v,s)=init_adam(params)
		else: (params,v,s)=update_params_adam(params,grad,lr,v,s,i)
		(cost_i,valid_per_i,correct_i)=g[-1](X[i],LAB[i],params,g,1)
		cost.append(cost_i)
		valid_per.append(valid_per_i)
		correct.append(correct_i)
		t2=time.time()
		print('%s bp detal time dt2= %0.2fs'%(i,t2-t1))
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
	valid_per=np.array(valid_per)
	correct=np.array(correct)
	lra=np.array(lra)
	t3=time.time()
	print('%s bp detal time dt3= %0.2fs'%(i,t3-t2))
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
#	plt.xlabel('Iterations *%s'%batch)
#		var_title=(lr0,klr,batch)
#		title='lr0=%.3e\n klr=%s\n batch=%s\n'%var_title
#	title='l2_grad'
#	plt.title(title,loc='left')
#	plt.show()
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
#		Xi=np.expand_dims(X[i],-1)
#		print('X%s.shape='%i,Xi.shape)
#		LABi=np.expand_dims(LAB[i],-1)
#def cross_entropy(X,LAB,params,g,isvalid=0):
		(cost_i,valid_per_i,correct_i)=g[-1](X[i],LAB[i],params,g,1)
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
#		Xi=np.expand_dims(X[i],-1)
		(cost_i,valid_per_i,correct_i)=g[-1](X[i],LAB[i],params,g,1)
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
	lab=mnist.test_lab[n].squeeze()
#	x=np.expand_dims(x,0)
#	lab=mnist.test_lab[n].squeeze()
	x=np.expand_dims(x,0)
	y=fp(x,params,g)
	y=np.argmax(y).squeeze()
	print('Real lab number is :\t%s'%lab)
	print('Precdict number is :\t%s'%y)
	img=x.reshape(28,28)
	plt.imshow(img,cmap='gray')
	plt.show()
## ==========

def train_and_valid(params,g,lr0=2e-3,klr=0.9995,batch=20,batches=0,isplot=0,istime=0,ischeck=0,isl2grad=0):
	(params,lrend)=batch_train(params,g,lr0,klr,batch,batches,isplot,istime,isl2grad=isl2grad)
	(valid_per,correct)=valid(params,g)
	(valid_per2,correct2)=valid_train(params,g)
	if ischeck==1:
		print('Grade check running ...')
		x=mnist.test_img[0]
		lab=mnist.test_lab[0]
		grad_check(x,lab,params,g)
		print('Grade check end.')
	return (lrend,valid_per,valid_per2)

def hyperparams_test(params,params_init,g,nloop=8,lr0=2e-3,klr=0.9995,batch=40,batches=0,isupdate=0,isl2grad=1):
	print('heyperparams_test: ...')
	print('heyperparams_test: layers =',int(len(params)/3)+1)
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
		(lrendi,v1,v2)=train_and_valid(params,g,lri,klr,batch,batches,isplot=1,isl2grad=isl2grad)
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


#ch=2
#nn=2
#lab=mnist.train_lab[nn:nn+imba]
#x=mnist.train_img[0:imba]
#x=np.expand_dims(x,1)
#x=np.random.randn(imba,1,imh,imw)
#x=np.random.randn(imba,1,imh,imw)*100
#i=1
#grad_check(x,lab,params,g)
#grad_check2(x,lab,params,g,pname='Y'+str(i),dv=1e-3)
#grad_check2(x,lab,params,g,pname='A'+str(i),dv=1e-3)
#grad_check2(x,lab,params,g,pname='M'+str(i),dv=1e-5)
#grad_check2(x,lab,params,g,pname='Z'+str(i),dv=1e-5)
#grad_check2(x,lab,params,g,pname='ZdM'+str(i),dv=1e-5)
#yfp,op=fp(x,params,g,isop=1)
#ybp,d_=bp(x,lab,params,g,isop=1)
#print('"d_[ZdM%s]=\n'%i,d_['ZdM'+str(i)].squeeze())
#print('op[ZdM%s]=\n'%i,op['ZdM'+str(i)].squeeze())
#print('op[Z%s]=\n'%i,op['Z'+str(i)].squeeze())


#print('params.keys()=\n',params.keys())
#for k,v in params.items(): print('%s.shape='%k,v.shape)
#print('op.keys()=\n',op.keys())
#print('d_.keys()=\n',d_.keys())


#def grad_check(x,lab,params,g,dv=1e-5):
#def fp(X,params,g,opin=None,pname=None,isop=0,e=1e-8):
#(params,params_init,g)=init_params()
#def batch_train(params,g,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1):
#(params,lre)=batch_train(params,g,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1)
#show(params,g)
#with open('cnn_p1.pkl', 'wb') as f: pickle.dump(params,f)
#with open('cnn_p1.pkl', 'rb') as f: params2=pickle.load(f)

