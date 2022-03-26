#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openpyxl
from matplotlib import pyplot as plt 
import numpy as np
import mnist



def pt(x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x
def convolution(a,b):
#    return  np.trace(a.dot(b.T))
    return (a*b).sum()

def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def d_tanh(x):
    y=tanh(x)
#    return np.diag(1/(np.cosh(x))**2)
#    return np.diag(1-y**2)
    return 1/(np.cosh(x))**2
#    return 1-y**2
def d_softmax(x):
    y=softmax(x)
    return np.diag(y)-np.outer(y,y)
#def dsoftmax(x):
#    sm=softmax(x)
#    k_softmax=np.diag(sm)-np.outer(sm,sm)
#    return dot(k_softmax,x)
#func=softmax
def plt_img(img):
    img=img.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()
#for i in range(n):
#    x=np.random.rand(n)
#    x=np.random.rand(n)
#    dev=k[func](x)
#    v1=func(x)
#    x[i]+=h
#    v2=func(x)
#    k_f=(v2-v1)/h
#print(params['b0'])
#print(params['w'])
#a=1*np.arange(12)
#b=10*np.arange(12)
#a=1*np.arange(12).reshape([3,4])
#b=10*np.arange(12).reshape([3,4])
#c=a*b
#print(a)
#print(b)
#print(a.shape)
#print(a.T.shape)
#print(np.transpose(a).shape)
##print(c.sum())
#print(a*b)
#print((a*b).sum())
#print(convolution(a,b))
#print(np.trace(a.dot(b.T)))
#print(np.dot(a.T,b.T))
#print(a.T.dot(b.T))
#print(np.outer(a,b))
#print(np.matmul(a.T,b))
#print(np.matmul(a,b))
#print(d_tanh(a))
#exit(0)

def loss(img,lab,params):
    ys=predict(img,params)
    yr=np.eye(ny)[lab]
    return (ys-yr).dot(ys-yr)
def predict(img,params):
    w=params['w']
    b1=params['b1']
    b0=params['b0']
    yi=tanh(img+b0)
    yl=w.dot(yi)+b1
    ys=softmax(yl)
    return ys
def grad_params(img,lab,params):
    w=params['w']     # m*n
    b1=params['b1'] # m*1
    b0=params['b0'] # n*1
    yi=tanh(img+b0)  # n*1
    yl=w.dot(yi)+b1 # m*1
    ys=softmax(yl) # m*1
    yr=np.eye(ny)[lab] # m*1
    l=(ys-yr).dot(ys-yr)  ##1*1
    d_l_ys = 2*(ys-yr)  ## 1*m
    d_ys_yl = d_softmax(yl) ## m*m
    d_yl_yi = w   ## m*n      
    d_yi_b0 = 1-(img+b0)**2 #  --m*n
    d_l_yl = d_l_ys.dot(d_ys_yl) # 1*m
    d_l_w = np.outer(d_l_yl,yi) # m*n
#    d_l_w = np.outer(yi,d_l_yl) # m*n
    d_l_b1=d_l_yl # 1*m
#    d_l_b0=d_l_yl.dot(d_yl_yi).dot(d_yi_b0) # 
    d_l_b0=d_l_yl.dot(d_yl_yi)*d_yi_b0 # 
    grad_w = -d_l_w
    grad_b1 = -d_l_b1
    grad_b0 = -d_l_b0
#    print(d_l_w)
#    print(d_l_b1)
#    print(d_l_b0)
#    print(d_l_w.shape)
#    print(d_l_b1.shape)
#    print(d_l_b0.shape)
#    print(yi.shape)
    return {'w':grad_w,'b1':grad_b1,'b0':grad_b0}
    ## ==========
    ## dl=a.dw.b => dl/dw = (a.T)*(b.T)_m*n
    ## ==========
    ## dl = dl/dys.dys/dyl.dyl = (dl/dys.dys/dyl).dw.yi = dl/dyl.dw.yi
    ## dl/dw = (dl/dyl.T)*(yi.T)_m*n

nx=28*28
ny=10
params_init={'b0':0.1*np.ones(nx),'b1':0.2*np.ones(ny),'w':0.1*np.ones([ny,nx])}
#num_img=1234
num_img=np.random.randint(50000)
img=mnist.train_img[num_img]
lab=mnist.train_lab[num_img]
params=params_init
pred=predict(img,params)
pred=pred+np.eye(len(pred))[lab]*0.001
l=loss(img,lab,params)
#print(pred)
#print(l)
#print(np.argmax(pred))
print("The Image is:\n %s"%np.argmax(pred))
#plt_img(img)

#h=1e-3
#grad_list=[]
#for i in range(10):
#    n=np.random.randint(50000)
#    test_param='b1'
#    params=params_init
#    grad_p= grad_params(mnist.train_img[n],mnist.train_lab[n],params)
#    derivative=grad_p[test_param][i]
#    v1=loss(mnist.train_img[n],mnist.train_lab[n],params)
#    params[test_param][i]+=h
#    v2=loss(mnist.train_img[n],mnist.train_lab[n],params)
#    grade=-(v2-v1)/h
#    delta=derivative-grade
#    grad_list.append(delta)
##    grad_list.append(derivative-(v2-v1)/h)
#    print('derivative is : ',derivative)
#    print('grade is : ',grade)
#    print('delta is : ',delta)
#print(np.abs(grad_list).max())
#exit(0)

#h=1e-3
#grad_list=[]
#for j in range(784):
#    for i in range(10):
#        n=np.random.randint(50000)
#        test_param='w'
#        params=params_init
#        grad_p = grad_params(mnist.train_img[n],mnist.train_lab[n],params)
#        derivative=grad_p[test_param][i][j]
##        print(derivative.shape)
#        print("derivative is : ",derivative)
#        v1=loss(mnist.train_img[n],mnist.train_lab[n],params)
#        params[test_param][i][j]+=h
#        v2=loss(mnist.train_img[n],mnist.train_lab[n],params)
#        slope=-(v2-v1)/h
#        print("slope is : " , slope)
#        delta=derivative-slope
#        print('delta is : ',delta)
#        grad_list.append(delta)
##        print(derivative-(v2-v1)/h)
#print(np.abs(grad_list).max())
#print(np.abs(grad_list).argmax())

#h=1e-3
#grad_list=[]
#for i in range(784):
#    n=np.random.randint(50000)
#    test_param='b0'
#    params=params_init
#    grad_p = grad_params(mnist.train_img[n],mnist.train_lab[n],params)
#    derivative=grad_p[test_param][i]
#    v1=loss(mnist.train_img[n],mnist.train_lab[n],params)
#    params[test_param][i]+=h
#    v2=loss(mnist.train_img[n],mnist.train_lab[n],params)
#    grade=-(v2-v1)/h
#    delta=derivative-grade
##    grad_list.append(derivative-(v2-v1)/h)
#    grad_list.append(delta)
#    print('derivative is : ',derivative)
#    print('grade is : ',grade)
##    print('delta is : ',delta)
##        print(derivative-(v2-v1)/h)
#print(np.abs(grad_list).max())
exit(0)

#a=range(1,10)
#b=range(1,4)
#x=np.diag(a)
#x=np.outer(a,b)
#one_hot=np.identity(10)
#pt(x)
#exit(0)

#a=np.arange(6).reshape(2,3)    
#a=np.arange(6)
#b=np.arange(12).reshape(3,4)
#c=np.arange(1,10)
#d=np.arange(1,13).reshape(4,3)
#b=np.arange(6)+1
print(a)
print(b)
print(c)
print(d)
#print(a*b)
#print(a.dot(b))
#print(a.dot(b))
#print(np.linalg.pinv(c))

#np.random.seed(1)
#img=list(range(784))
#img=np.random.randint(10,100,size=(2,4))
#x=np.random.randint(10,100,size=(4,1))
#x=np.random.randint(0,255,size=(nx))
#b0=np.random.randint(0,255,size=(nx))
#w=np.random.randint(-99,99,size=(ny,nx))
#b1=np.random.randint(-99,99,size=(ny))
#yr=np.random.randint(0,1,size=(ny))
#yi=tanh(x-b0)
#y2=dot(w1,y1)+b1
#yl=w.dot(yi)+b1
#ys=softmax(yl)
#yr=np.eye(10)[n]
#l=(ys-yr).dot(ys-yr)
#pt(b0)
#pt(w1)
#pt(b1)
#pt(x)
#pt(y1)
#pt(y2)
#pt(yo)
#pt(yr)
#pt(l)
#k_l_ys=2*(ys-yr)
#k_ys_yl=k_sm(yl)
#def k_yl_w(i,j):
#    k_yl_w=np.eye(ny)[i]*yi[j]
#    return k_yl_w
#k_l_w=k_l_yo.dot(k_yo_y2).dot(k_y2_w)
#k_l_w=np.outer(k_l_yo.dot(k_yo_y2),k_y2_w)
#k_l_w=k_yo_y2.dot(k_l_yo).dot(k_y2_w)
#k_l_yl=k_l_ys.dot(k_ys_yl)
#k_l_wij=k_l_ys.dot(k_ys_yl).dot(k_yl_w(2,1))
#k_l_wij=k_l_ys.dot(k_ys_yl).dot( np.eye(ny)[3]*yi[1] )
#k_l_b1i=k_l_ys.dot(k_ys_yl).dot(np.eye(ny)[3])
#k_l_b0i=0   
#pt(k_l_wij)
#pt(k_l_b1i)
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

#x=k_tanh(range(4))
#pt(x)
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
















