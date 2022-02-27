#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openpyxl
from matplotlib import pyplot as plt 
#import wbsta
#import numpy as np
x=0


i=0
db=[[None for i in range(9)]for i in range(5)]
#db=[[]for i in range(5)]
for c in range(5):
	for r in range(9):
		i=i+1
		db[c][r]=i


x=db
print(type(x))
print(dir(x))
print(x)


