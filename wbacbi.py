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

wbn1=wb1.get_sheet_names()#获取sheet页
wbs1=wb1.get_sheet_by_name(wbn1[-5])
wbn2=wb2.get_sheet_names()
wbs2=wb2.get_sheet_by_name(wbn2[0])

maxr=wbs1.max_row    #最大行数
maxc=wbs1.max_column  #最大列数


#db1=[]
db1=[["" for i in range(1+maxr)]for i in range(1+maxc)]
#val1=''
for c in range(1,1+maxc):
	for r in range(1,1+maxr):
		val = wbs1.cell(row=r,column=c).value
#		if val is not None
#        db1[c][r] = wbs1.cell(row=r, column=c).value
#		db1[c][r] = [val if val is not None else ""]
		db1[c][r] = (val if val is not None else '')


for c in range(1,1+maxc):
	for r in range(1,1+maxr):
		wbs2.cell(row=r,column=c).value=db1[c][r]
#		wbs2.cell(row=r,column=c).value=str(db1[c][r])
#		wbs2.cell(row=r,column=c)=db1[c][r]

#x=str(db1[1][1])
x=db1[1][1]
#x=db1[1]
wbs2.cell(row=1,column=1).value=x
wb2.save('test2.xlsx')

print("dir x :\n ",dir(x))
print("x type is:\n ",type(x))
print("x is:\n ",x)
exit

