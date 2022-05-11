#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
#import json
#import pandas as pd
from matplotlib import pyplot as plt 

#(m,n)=(2,2)
(ba,m,n)=(2,3,3)
e=1e-8
np.random.seed(0)
#z=np.random.rand(m,m)
z=np.arange(ba*m*n).reshape(ba,m,n)*1e-3
u=np.mean(z,axis=(1,2),keepdims=1)
v=np.var(z,axis=(1,2),keepdims=1)
#u=np.mean(z,axis=(1,2))

def grad_im(z,e=1e-8):
	u=np.mean(z,axis=(1,2),keepdims=1)
	v=np.var(z,axis=(1,2),keepdims=1)
	x=(z-u)/(v+e)**0.5
	xx=np.einsum('mij,mkl->mijkl',x,x)
	Im=np.ones((xx.shape))
	mE=np.zeros((xx.shape))
	np.einsum('mijij->mij',mE)[:]=mE.shape[1]*mE.shape[2]
	
	

#v=np.var(z,axis=(1,2))
#u=np.mean(z)
#v=np.var(z)
x=(z-u)/(v+e)**0.5
mz=x.shape[1]
nz=x.shape[2]
XX=np.einsum('mij,mkl->mijkl',x,x)
Imn=np.ones((XX.shape))
mnE=np.zeros((XX.shape))
np.einsum('mijij->mij',mnE)[:]=mz*nz
#dXi_Zi=(mi*np.eye(mi)-np.ones((mi,mi))-XX)/(mi*(vi+e)**0.5)
#dXi_Zi=(mnE-Imn-XX)/(mz*nz*(v+e)**0.5)
vre=v.reshape((-1,)+tuple([1]*(XX.ndim-1)))
dXi_Zi=(mnE-Imn-XX)/(mz*nz*(vre+e)**0.5)
grad=dXi_Zi
#grad2=(m1c*np.eye(m1c)-np.ones((m1c,m1c))-x1c@x1c.T)/(m1c*(v1c+e)**0.5)
grad2=np.zeros((grad.shape))
for i in range(len(x)):
	xx=np.einsum('ij,kl->ijkl',x[i],x[i])
	Imn=np.ones((xx.shape))
	mnE=np.zeros((xx.shape))
	(m,n)=x[i].shape
	np.einsum('ijij->ij',mnE)[:]=m*n
	vi=np.squeeze(v[i])
	grad2[i]=(mnE-Imn-xx)/(m*n*(vi+e)**0.5)
#	grad2=(m1c*np.eye(m1c)-np.ones((m1c,m1c))-x1c@x1c.T)/(m1c*(v1c+e)**0.5)

zr=z.reshape(ba,-1,1)
ur=np.mean(zr,axis=1,keepdims=1)
vr=np.var(zr,axis=1,keepdims=1)
#xr=(zr-ur)/(vr+e)**0.5
#XXr=np.einsum('mij,mkj->mik',xr,xr)
xr=(zr-ur)/(vr+e)**0.5
XXr=np.einsum('mij,mkj->mik',xr,xr)
#XXr=np.einsum('mi,mj->mij',xr,xr)
print('XXr.shape=',XXr.shape)
print('XXr=\n',XXr)
(ba,mr,nr)=XXr.shape
Imnr=np.ones((XXr.shape))
#mnErd=np.eye(mr*nr)*mr*nr
#mnEr=np.r_[mnErd,mnErd]
mnEr=np.zeros((XXr.shape))
np.einsum('mii->mi',mnEr)[:]=mr
#vre=v.reshape((-1,)+tuple([1]*(XX.ndim-1)))
dXi_Zir=(mnEr-Imnr-XXr)/(mr*(vr+e)**0.5)
grad3=dXi_Zir


#zcopy=copy.deepcopy(z)
#dv=1e-5
#k=np.zeros((grad.shape))
#for i in range(k.shape[0]):
#	for j in range(k.shape[1]):
#		for r in range(k.shape[2]):
#			for c in range(k.shape[3]):
##		for r in range(len(z)):
##			for c in range(len(z[r])):
#				vbak=z[r,c]
##				vbak=copy.deepcopy(z[r,c])
#				z[r,c]=vbak-dv
#				u=np.mean(z)
#				v=np.var(z)
##				print('z1=\n',z)
#				x1=(z[r,c]-u)/(v+e)**0.5
#				z[r,c]=vbak+2*dv
#				u=np.mean(z)
#				v=np.var(z)
##				print('z2=\n',z)
#				x2=(z[r,c]-u)/(v+e)**0.5
#				z[r,c]=vbak
#				ktmp=(x2-x1)/(2*dv)
#				k[i,j,r,c]=ktmp
#assert(np.all(z)==np.all(zcopy))

#print('u=',u)
#print('v=',v)
#print('z=\n',z)
#print('x=\n',x)
#print('xx=\n',xx)
#print('Imn=\n',Imn)
#print('mnE=\n',mnE)
print('grad=\n',grad.reshape(1,-1))
print('grad2=\n',grad2.reshape(1,-1))
print('grad3=\n',grad3.reshape(1,-1))
print('delta_grad=\n',grad.reshape(ba,-1)-grad2.reshape(ba,-1))
print('delta_grad3=\n',grad3.reshape(ba,-1)-grad2.reshape(ba,-1))
#print('k=\n',k.reshape(1,-1))
#print('xx.shape=',xx.shape)
#print('Imn.shape=',Imn.shape)
#print('mnE.shape=',mnE.shape)
#print('k.shape=',k.shape)
print('grad.shape=',grad.shape)
print('grad2.shape=',grad2.shape)
print('grad3.shape=',grad3.shape)



