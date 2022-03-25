#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openpyxl
from matplotlib import pyplot as plt 
import numpy as np

def pt(x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x
#def dot(x1,x2):
#    return np.dot(x1,x2)
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def k_tanh(x):
    y=tanh(x)
#    return np.diag(1/(np.cosh(x))**2)
    return np.diag(1-y**2)
def k_sm(x):
    y=softmax(x)
    return np.diag(y)-np.outer(y,y)
def dsoftmax(x):
    sm=softmax(x)
    k_softmax=np.diag(sm)-np.outer(sm,sm)
    return dot(k_softmax,x)
#pt(k_tanh(x))
#pt(k_sm(x))
k={softmax:k_sm,tanh:k_tanh}
#func=softmax
func=tanh
n=4
h=0.001
for i in range(n):
#    x=np.random.rand(n)
    x=np.random.rand(n)
    dev=k[func](x)
    v1=func(x)
    x[i]+=h
    v2=func(x)
    k_f=(v2-v1)/h
#    k_f=(func(x+h)-func(x))/h
#    x2=x*2+h
#    print(x)
#    print(x2)
#    print(func(x))
#    print(func(x2))
#    print(func(x+100000))
#    print('===')
#    print(dev[i])
#    print(k_f)
#    print(dev[i]-k_f)

ny=10
def loss(img,lab,params):
    ys=predict(img,params)
    yr=np.eye(ny)[lab]
    return (ys-yr).dot(ys-yr)

loss(0,1,0)
exit(0)

#a=range(1,10)
#b=range(1,4)
#x=np.diag(a)
#x=np.outer(a,b)
#one_hot=np.identity(10)
#pt(x)
#exit(0)

ny=10
nx=12
n=3
np.random.seed(1)
#img=list(range(784))
#img=np.random.randint(10,100,size=(2,4))
#x=np.random.randint(10,100,size=(4,1))
x=np.random.randint(0,255,size=(nx))
b0=np.random.randint(0,255,size=(nx))
w=np.random.randint(-99,99,size=(ny,nx))
b1=np.random.randint(-99,99,size=(ny))
yr=np.random.randint(0,1,size=(ny))
yi=tanh(x-b0)
#y2=dot(w1,y1)+b1
yl=w.dot(yi)+b1
ys=softmax(yl)
yr=np.eye(10)[n]
l=(ys-yr).dot(ys-yr)
#pt(b0)
#pt(w1)
#pt(b1)
#pt(x)
#pt(y1)
#pt(y2)
#pt(yo)
#pt(yr)
#pt(l)
k_l_ys=2*(ys-yr)
k_ys_yl=k_sm(yl)
def k_yl_w(i,j):
    k_yl_w=np.eye(ny)[i]*yi[j]
    return k_yl_w
#k_l_w=k_l_yo.dot(k_yo_y2).dot(k_y2_w)
#k_l_w=np.outer(k_l_yo.dot(k_yo_y2),k_y2_w)
#k_l_w=k_yo_y2.dot(k_l_yo).dot(k_y2_w)
k_l_yl=k_l_ys.dot(k_ys_yl)
#k_l_wij=k_l_ys.dot(k_ys_yl).dot(k_yl_w(2,1))
k_l_wij=k_l_ys.dot(k_ys_yl).dot( np.eye(ny)[3]*yi[1] )
k_l_b1i=k_l_ys.dot(k_ys_yl).dot(np.eye(ny)[3])
k_l_b0i=0   
pt(k_l_wij)
pt(k_l_b1i)
#k_l_w=np.outer(yi,k_l_yl)
#k_l_w=k_l_ys.dot(k_ys_yl).dot(k_yl_w)
#pt(k_l_ys)
#pt(k_ys_yl)
#pt(k_l_yl)
#pt(k_l_yl.shape)
#pt(k_yl_w(2,1).T)
#pt(k_yl_w(2,1))
#pt(k_yl_w(2,1).T.shape)
#pt(np.eye(ny)[2]*yi[1])
#pt(yi)

x=k_tanh(range(4))
pt(x)
#one_hot=np.identity(10)
#one_hot=np.eye(10)
#pt(one_hot[5])
#yr=one_hot[n]
#db1=
#k_y2_b1=np.ones(len(b1))
#dy2=dot(k_y2_b1,db1)
#dyo=dot(k_softmax(y2),dy2)
#dyo=dsoftmax(y2)
#l=dot(yo-yr,yo-yr)
#dl=dot(dyo,2*(yo-yr))
#k_l_b1=dot()
#pt(dl)
exit(0)


















