#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import openpyxl
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 


def pt(x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x

file='dt1.xlsx'
dfn5=pd.read_excel(file,sheet_name=-5)
dfn6=pd.read_excel(file,sheet_name=-6)
df1=dfn6.iloc[51:92,6:]
df2=dfn5.iloc[51:92,6:]
dfp=dfn5.iloc[48-2:51-1,7-1:]
dfd=df2
for x in range(df1.shape[0]):
    for y in range(df1.shape[1]):
        d1=df1.iloc[x,y]
        d2=df2.iloc[x,y]
#        if d1!=0 : v=d2/d1;  v='%.3f%%'%v; dfd.iloc[x,y]=v
#        if d1!=0 : dfd.iloc[x,y]='%.1f%%'%(d2/d1)
        if d1!=0 : dfd.iloc[x,y]='%.1f%%'%(d2/d1*100)
#        if d1!=0 : dfd.iloc[x,y]=d2/d1
#pt(dfd.columns)
#pt(dfd.iloc[10,10])
#        if d1!=0 : v=float(d2)/float(d1)
#           
#        if d1!=0 :  dfd.iloc[x,y]=d2/d1
#        if d1!=0 :  dfd.iloc[x,y]='%.3f%%'%(d2/d1)
dfdmax=dfd.max().to_frame().T
#pt(dfdmax)
#exit(0)
dfdmax=dfd.max().to_frame().T
dfdmin=dfd.min().to_frame().T
dfdmean=dfd.mean().to_frame().T
dfdstd=dfd.std().to_frame().T
dfa=pd.concat([dfd,dfp,dfdmean,dfdstd,dfdmin,dfdmax])
#dfa=dfd
dfa.to_excel('tmp.xlsx')

