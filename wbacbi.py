#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openpyxl
import numpy as np 
from matplotlib import pyplot as plt 

#class wbacbi:
#    def __init__(self,wbfile):
#        self.wbfile=wbfile
#        self.db=[]
#        self.nrowsPreData=5
#        self.unitIndex=1
#        self.wb2db(wbfile)

#    def wb2db(self,wbfile):
#        wb=openpyxl.load_workbook(wbfile)




## ==========
x=0
#wbfile='data1.xlsx'
#wbfile='/home/qc/download/2201-06-PM_VA05A_HTDR_Result_500hrs_20220216.xlsm'
wbfile='test.xlsm'
wb1=openpyxl.load_workbook(wbfile)
wb2=openpyxl.Workbook()

sheets1=wb1.get_sheet_names()#获取sheet页
sheet1=wb1.get_sheet_by_name(sheets1[-5])
sheets2=wb2.get_sheet_names()
sheet2=wb2.get_sheet_by_name(sheets2[0])
sheet3=wb2.get_sheet_by_name(sheets2[1])

max_row=sheet1.max_row#最大行数
max_column=sheet1.max_column#最大列数

for r in range(1, 1+51):
    for c in range(1, 1+max_column):
        sheet2.cell(row=r, column=c).value = sheet1.cell(row=r, column=c).value


sheet1a=wb1.get_sheet_by_name(sheets1[-5])
sheet1b=wb1.get_sheet_by_name(sheets1[-6])
a=sheet1a.cell(row=53, column=7).value
b=sheet1b.cell(row=53, column=7).value
#sheet2.cell(row=53, column=7).value = a/b
sheet2.cell(row=53, column=7).value = '{:.2%}'.format(a/b)

#db=[]
d3=0
#max_column=7+5
#max_row=53+10
#db=[[0 for i in range(53,1+max_row)]for i in range(7,1+max_column)]
db=[[None for i in range(1+max_row)]for i in range(1+max_column)]
for c in range(7, 1+max_column):
	for r in range(53, 1+max_row):
		d1=sheet1a.cell(row=r,column=c).value
		d2=sheet1b.cell(row=r,column=c).value
		if d1 is not None and d2 is not None :
			if d2 !=0 :
				d3 = '{:.2%}'.format(d1/d2) 
				sheet2.cell(row=r, column=c).value = d3 
#				print(d3)
				db[c][r] = d3
#				db[c-7][r-53].append(d3)




#                sheet2.cell(row=r, column=c).value = data 




wb2.save('test2.xlsx')
x=db
#x=q.db
#x=q.wb2db(wbfile)

## ==========

## ==========


print(dir(x))
print(type(x))
print(x)


wbfile='data2.xlsx'
