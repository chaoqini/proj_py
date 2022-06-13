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
(lays,imba,imchin,dimch,imh,imw,convk,km,minhw)=(5,2,2,2,28,28,3,2,10)
#(lays,imba,imchin,dimch,imh,imw,convk,km,minhw)=(5,2,2,2,8,8,3,2,8)
np.random.seed(0)
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

#def im2col_idt(im,k=3,s=1,pad=1):
##		if k==1: col=np.expand_dims(im,(-2,-1))
##		else:
#			p=int(k/2)
#			(ba,c,h,w)=im.shape
#			if pad==1: imp=np.pad(im,((0,0),(0,0),(p,p),(p,p)))
#			else: imp=im
#			(ba,c,hp,wp)=imp.shape
#			strd=(c*hp*wp,hp*wp,wp*s,s,wp,1)
#			strd=(i*imp.itemsize for i in strd)
#			col=np.lib.stride_tricks.as_strided(imp,shape=(ba,c,int((hp-k)/s)+1,int((wp-k)/s)+1,k,k),strides=strd)
#			im_idt=np.expand_dims(im,(-2,-1))
#			col=col+im_idt
#    return col


#def init_params(lays=lays,k=convk,nh=imh,nw=imw,nk=2,ny=10,func=0,seed=0):
def init_params(lays=lays,k=convk,imchin=imchin,dimch=dimch,imh=imh,imw=imw,ch=-1,func=None):
	if func==None: (func,func_d)=(relu,relu_d)
	if ch==-1:
		ch=[1]
		for i in range(lays-1):ch.append(imchin+dimch*i)
		ch.append(10)
	(params_init,g,g_d,l2_grad)=({},[],[],{})
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
		g_d.append(func_d)
	g[-1]=softmax;g.append(cross_entropy)
	params=copy.deepcopy(params_init)
#	for k,v in params.items():
#		print('params: %s.shape='%k,v.shape)
	return (params,params_init,g,g_d)
(params,params_init,g,g_d)=init_params()

def maxpooling(z,k=2,e=1e-12):
	mkk=im2col(z,k=2,s=2,pad=0)
	m=np.max(mkk,(-2,-1))
	zdm=(mkk+1-mkk.max((-2,-1),keepdims=1))
	zdm[zdm<1-e]=0
	return m,zdm
def maxpooling_d(d_m,zdm):
	(ba,c,hp,wp,khp,kwp)=zdm.shape
	d_z_mkk=np.expand_dims(d_m,(-2,-1))*zdm
	strd=(c*hp*khp*wp*kwp,hp*khp*wp*kwp,wp*kwp,1)
	strd=(i*d_z_mkk.itemsize for i in strd)
	d_z=np.lib.stride_tricks.as_strided(d_z_mkk,shape=(ba,c,hp*khp,wp*kwp),strides=strd)
	return d_z
## ==========
def fp(X,params,g,opin=None,pname=None,isop=0,e=1e-8):
	ba=X.shape[0]
	(l,OP)=(int(len(params)/3)+1,{})
	OP['A-1']=X
	for i in range(l) :
		Ai_1=OP['A'+str(i-1)]
		if i==l-1:
			wi=params['w'+str(i)]
			Zi=np.einsum('bchw,ochw->bo',Ai_1,wi)
			if pname=='Z'+str(i): Zi=opin
			Yi=np.expand_dims(Zi,(1,-1))
			if pname=='Y'+str(i): Yi=opin
		else:
			ki=params['k'+str(i)]
			Ci_1=im2col(Ai_1,ki.shape[-1])
			if pname=='C'+str(i-1): Ci_1=opin
			Zi_tmp=np.einsum('bchwij,mcij->bmchw',Ci_1,ki)
			Ai_1_idt=np.expand_dims(Ai_1,1)
			Zi_tmp=Zi_tmp+Ai_1_idt
			Zi=np.einsum('bmchw->bmhw',Zi_tmp)
			if pname=='Z'+str(i) : Zi=opin
			if Zi.shape[-1]>=km*minhw and Zi.shape[-2]>=km*minhw: 
				Mi,ZdMi=maxpooling(Zi,km)
			else:
				Mi=Zi
			if pname=='M'+str(i) : Mi=opin
			if pname=='ZdM'+str(i) : ZdMi=opin
			ui=Mi.mean((-2,-1),keepdims=1)
			vi=Mi.var((-2,-1),keepdims=1)
			Xi=(Mi-ui)/(vi+e)**0.5
			if pname=='X'+str(i): Xi=opin
			gamai=params['gama'+str(i)]
			betai=params['beta'+str(i)]
			Yi=gamai*Xi+betai
			if pname=='Y'+str(i): Yi=opin
		Ai=g[i](Yi)
		if pname=='A'+str(i): Ai=opin
		OP['C'+str(i-1)]=Ci_1
		OP['Z'+str(i)]=Zi
		OP['ZdM'+str(i)]=ZdMi
		OP['M'+str(i)]=Mi
		OP['u'+str(i)]=ui
		OP['v'+str(i)]=vi
		OP['X'+str(i)]=Xi
		OP['Y'+str(i)]=Yi
		OP['A'+str(i)]=Ai
	Y=OP['A'+str(l-1)]
	if isop==0: 
		return Y
	else: 
		return (Y,OP)

## ==========
def bp(X,LAB,params,g,g_d,e=1e-8,isop=0):
	(Y,OP)=fp(X,params,g,isop=1)
	ba=Y.shape[0]
	YL=np.zeros(Y.shape)
	YL[np.arange(ba),0,LAB.reshape(ba),0]=1
	(l,d_,grad)=(int(len(params)/3)+1,{},{})
	for i in range(l-1,-1,-1):
		if i==l-1: 
			wi=params['w'+str(i)]
			Ai_1=OP['A'+str(i-1)]
			d_Yi=Y-YL
			d_Ai_1=np.einsum('bpoq,ochw->bchw',d_Yi,wi)
			d_wi=np.einsum('bchw,bpoq->bochw',Ai_1,d_Yi)
			grad['d_w'+str(i)]=d_wi.mean(0)
			d_['Y'+str(i)]=d_Yi
		else:
			ki=params['k'+str(i)]
			gamai=params['gama'+str(i)]
			betai=params['beta'+str(i)]
			ui=OP['u'+str(i)]
			vi=OP['v'+str(i)]
			Xi=OP['X'+str(i)]
			Ci_1=OP['C'+str(i-1)]
			d_Yi=d_['Y'+str(i)]
			d_Xi=gamai*d_Yi
			d_gamai=(d_Yi*Xi).sum((-2,-1),keepdims=1)
			d_betai=(d_Yi).sum((-2,-1),keepdims=1)
			grad['d_gama'+str(i)]=d_gamai.mean(0)
			grad['d_beta'+str(i)]=d_betai.mean(0)
			XX=np.einsum('bcij,bckl->bcijkl',Xi,Xi)
			Imm=np.ones((XX.shape))
			mmE=np.zeros((XX.shape))
			np.einsum('bcijij->bcij',mmE)[:]=mmE.shape[-2]*mmE.shape[-1]
			vi=np.expand_dims(vi,(-2,-1))
			dXi_Mi=(mmE-Imm-XX)/(mmE.shape[-2]*mmE.shape[-1]*(vi+e)**0.5)
			d_Mi=np.einsum('bcijkl,bckl->bcij',dXi_Mi,d_Xi)
			ZdMi=OP['ZdM'+str(i)]
			Zi=OP['Z'+str(i)]
			if Zi.shape[-1]>=km*minhw and Zi.shape[-2]>=km*minhw: 
				d_Zi=maxpooling_d(d_Mi,ZdMi)
			else:
				d_Zi=d_Mi
			ki_fl=np.flip(ki,(-2,-1))
			d_Zi_2col=im2col(d_Zi,ki_fl.shape[-1])
			d_Ai_1_tmp=np.einsum('bmhwij,mcij->bchw',d_Zi_2col,ki_fl)
			d_Zi_m2c=d_Zi.sum(1,keepdims=1)
#			d_Zi_m2c=np.einsum('bmhw->bhw',d_Zi)
#			d_Zi_m2c=np.expand_dims(d_Zi_m2c,1)
			d_Ai_1=d_Ai_1_tmp+d_Zi_m2c
			d_Zi=np.pad(d_Zi,((0,0),(0,0),(0,Ci_1.shape[2]-d_Zi.shape[2]),(0,Ci_1.shape[3]-d_Zi.shape[3])))
			d_ki=np.einsum('bchwij,bmhw->bmcij',Ci_1,d_Zi)
			grad['d_k'+str(i)]=d_ki.mean(0)
			if isop!=0:
				d_['X'+str(i)]=d_Xi
				d_['M'+str(i)]=d_Mi
				d_['ZdM'+str(i)]=ZdMi
				d_['Z'+str(i)]=d_Zi
				d_['Z'+str(i)+'_2col']=d_Zi_2col
				d_['A'+str(i-1)]=d_Ai_1
		if i>=1:
			Yi_1=OP['Y'+str(i-1)]
			d_Ai_1=np.pad(d_Ai_1,((0,0),(0,0),(0,Yi_1.shape[2]-d_Ai_1.shape[2]),(0,Yi_1.shape[3]-d_Ai_1.shape[3])))
			d_Yi_1=g_d[i-1](Yi_1)*d_Ai_1
			d_['Y'+str(i-1)]=d_Yi_1
	if isop!=0: 
		return (grad,d_)
	else: 
		return grad
	
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
#			print('x.shape=',x.shape)
			l1=g[-1](x,lab,pt,g)
#			print('l1=\n',l1)
			i[...]=vbak+dv
			l2=g[-1](x,lab,pt,g)
#			print('l2=\n',l2)
#			kk=(l2-l1)/(2*dv)
			kk=(l2-l1)/(2*dv)
#			if (v.size<200 and nloop%10==0) or nloop%100==0 : 
			if nloop%(int(nloop/4)+1) ==0 : 
				print('slope: %s[%s/%s] slope = %s'%(k,nloop,v.size,kk))
#			kk=kk.mean(0)
			slp['d_'+k].append(kk)
			i[...]=vbak
		slp['d_'+k]=np.array(slp['d_'+k]).reshape(v.shape)
#	print('pt=',pt)
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
	v=op[pname]
	slp['d_'+pname]=[]
	for i in np.nditer(v,op_flags=['readwrite']):
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
def grad_check(x,lab,params,g,g_d,dv=1e-5):
	y1=bp(x,lab,params,g,g_d)
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
		if v.size<200:
			print('=== grad_check : ')
			print('grad[%s].shape='%k,y1[k].shape)
			print('slope[%s].shape='%k,y2[k].shape)
			print('grad[%s]=\n'%k,y1[k].squeeze())
			print('slope[%s]=\n'%k,y2[k].squeeze())
	print('grad_check: abs_error=\n',abs_error)
	print('grad_check: ratio_error=\n',ratio_error)
	return (y1,y2)
def grad_check2(x,lab,params,g,g_d,pname=None,dv=1e-5):
	check_slp=slope2(x,lab,params,g,pname=pname,dv=dv)
	y1,d_=bp(x,lab,params,g,g_d,isop=1)
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

def batch_train(params,g,g_d,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1):
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
		pn=i%(max(int(len(X)/10),1))
		if pn==0 or i==len(X)-1:
			print('Training iteration number = %s/%s'%(i,len(X)))
#		Xi=np.expand_dims(X[i],-1)
#		LABi=np.expand_dims(LAB[i],-1)
		grad=bp(X[i],LAB[i],params,g,g_d)
		lr=lr0*klr**i
		lra.append(lr)
		if i==0: (v,s)=init_adam(params)
		else: (params,v,s)=update_params_adam(params,grad,lr,v,s,i)
		(cost_i,valid_per_i,correct_i)=g[-1](X[i],LAB[i],params,g,1)
		cost.append(cost_i)
		valid_per.append(valid_per_i)
		correct.append(correct_i)
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


#ch=2
#nn=2
#lab=mnist.train_lab[nn:nn+imba]
#x=mnist.train_img[0:imba]
#x=np.expand_dims(x,1)
#x=np.random.randn(imba,1,imh,imw)
#x=np.random.randn(imba,1,imh,imw)*100
#i=2
#grad_check(x,lab,params,g,g_d)
#grad_check2(x,lab,params,g,g_d,pname='M'+str(i),dv=1e-5)
#grad_check2(x,lab,params,g,g_d,pname='Z'+str(i),dv=1e-5)
#grad_check2(x,lab,params,g,g_d,pname='ZdM'+str(i),dv=1e-5)
#yfp,op=fp(x,params,g,isop=1)
#ybp,d_=bp(x,lab,params,g,g_d,isop=1)
#print('"d_[ZdM%s]=\n'%i,d_['ZdM'+str(i)].squeeze())
#print('op[ZdM%s]=\n'%i,op['ZdM'+str(i)].squeeze())
#print('op[Z%s]=\n'%i,op['Z'+str(i)].squeeze())


#print('params.keys()=\n',params.keys())
#for k,v in params.items(): print('%s.shape='%k,v.shape)
#print('op.keys()=\n',op.keys())
#print('d_.keys()=\n',d_.keys())


#def grad_check(x,lab,params,g,g_d,dv=1e-5):
#def fp(X,params,g,opin=None,pname=None,isop=0,e=1e-8):
#(params,params_init,g,g_d)=init_params()
#def batch_train(params,g,g_d,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1):
#(params,lre)=batch_train(params,g,g_d,lr0=2e-3,klr=0.9995,batch=32,batches=0,isplot=0,istime=0,isl2grad=1)
#show(params,g)
#with open('cnn_p1.pkl', 'wb') as f: pickle.dump(params,f)
#with open('cnn_p1.pkl', 'rb') as f: params2=pickle.load(f)

