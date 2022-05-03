#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mnist
import copy
#import json
#import pandas as pd
from matplotlib import pyplot as plt 

m=2
e=1e-8
np.random.seed(0)
z=np.random.rand(m,m)
u=np.mean(z)
v=np.var(z)
x=(z-u)/(v+e)**0.5

xx=np.einsum('ij,kl->ijkl',x,x)
Im=np.ones((xx.shape))
#Im=np.ones((xx.shape),dtype=np.float32)
me=np.zeros((xx.shape))
np.einsum('ijij->ij',me)[:]=1
tt=xx+Im
grad=(me-Im-xx)/(m*(v+e)**0.5)

print('u=',u)
print('v=',v)
print('z=\n',z)
print('x=\n',x)

print('xx=\n',xx)
print('Im=\n',Im)
print('tt=\n',Im)
print('me=\n',me)
print('grad=\n',grad)

print('xx.shape=',xx.shape)
print('Im.shape=',Im.shape)
print('tt.shape=',tt.shape)
print('me.shape=',me.shape)
print('grad.shape=',grad.shape)

dv=1e-6
k=np.zeros((grad.shape))
for r in range(len(z)):
	for c in range(len(z[r])):
#		vb=z[r,c]
		z1=z[r,c]-dv
		x1=(z1-u)/(v+e)**0.5
		z2=z[r,c]+dv
		x2=(z2-u)/(v+e)**0.5
		ktmp=(x2-x1)/(2*dv)
		k[r,c,r,c]=ktmp


print('grad=\n',grad)
print('k=\n',k)
print('grad.shape=',grad.shape)
print('k.shape=',k.shape)



