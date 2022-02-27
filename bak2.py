#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openpyxl
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
sheets2=wb2.get_sheet_names()
sheet1=wb1.get_sheet_by_name(sheets1[-5])
sheet2=wb2.get_sheet_by_name(sheets2[0])

max_row=sheet1.max_row#最大行数
max_column=sheet1.max_column#最大列数
#for r in range(1,1+51):
#    for c in range(97,97+max_column):#chr(97)='a'
#	n=chr(n)#ASCII字符
#	i='%s%d'%(n,m)#单元格编号
#	cell1=sheet1[i].value#获取data单元格数据
#	sheet2[i].value=cell1#赋值到test单元格

for r in range(1, 1+51):
    for c in range(1, 1+max_column):
        sheet2.cell(row=r, column=c).value = sheet1.cell(row=r, column=c).value


sheet1a=wb1.get_sheet_by_name(sheets1[-5])
sheet1b=wb1.get_sheet_by_name(sheets1[-6])
a=sheet1a.cell(row=53, column=7).value
b=sheet1b.cell(row=53, column=7).value
sheet2.cell(row=53, column=7).value = a/b





wb2.save('test2.xlsx')
x=wb2
#x=q.db
#x=q.wb2db(wbfile)

## ==========

## ==========


print(dir(x))
print(type(x))
print(x)


wbfile='data2.xlsx'
