#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy

def pt(self,x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x


pt(111)
