#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np 
import pandas as pd
#from matplotlib import pyplot as plt 

class wbqual:
    def __init__(self,file):
        self.file=file

    def pt(self,x,display=1,fexit=0):
        print("dir is:\n",dir(x))
        print("type is:\n ",type(x))
        if display==1: print("variable is:\n",x)
        if fexit!=0: exit(0)
        return x

    def makeratio(self,target='tmp.xlsx'):
        dfn5=pd.read_excel(self.file,sheet_name=-5)
        dfn6=pd.read_excel(self.file,sheet_name=-6)
        df1=dfn6.iloc[51:92,6:]
        df2=dfn5.iloc[51:92,6:]
        dfp=dfn5.iloc[48-2:51-1,7-1:]
        dfd=df2
        for x in range(df1.shape[0]):
            for y in range(df1.shape[1]):
                d1=df1.iloc[x,y]
                d2=df2.iloc[x,y]
#        if d1!=0 : dfd.iloc[x,y]='%.1f%%'%(d2/d1*100)
                if d1!=0 : dfd.iloc[x,y]=d2/d1
        dfdmax=dfd.max().apply(lambda x : '%.2f%%'%(x*100) if type(x)==float else '' ).to_frame().T
        dfdmin=dfd.min().apply(lambda x : '%.2f%%'%(x*100) if type(x)==float else '' ).to_frame().T
        dfdmean=dfd.mean().apply(lambda x :'%.2f%%'%(x*100)).to_frame().T
        dfd4sigmaper=(dfd.std()/dfd.mean()*4).apply(lambda x :'%.2f%%'%(x*100)).to_frame().T
        dfa=pd.concat([dfd,dfp,dfdmean,dfd4sigmaper,dfdmin,dfdmax])
        dfa.to_excel(target)

def main(file=sys.argv[1],target='tmp.xlsx'):
    q=wbqual(file)
    if len(sys.argv)>2: target=sys.argv[2]
    q.makeratio(target)
if __name__ == '__main__' : main()   
#if __name__ == '__main__' : main(sys.argv[1],sys.argv[2])   
