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
wbfile1='test.xlsm'
wb1=openpyxl.load_workbook(wbfile1)
wbfile2='test2.xlsx'
wb2=openpyxl.load_workbook(wbfile2)
#wb2=openpyxl.Workbook()

#wbn1=wb1.get_sheet_names()#获取sheet页
#wbs1=wb1.get_sheet_by_name(wbn1[-5])
#wbn2=wb2.get_sheet_names()
#wbs2=wb2.get_sheet_by_name(wbn2[0])

wb1s1=wb1.worksheets[0]
wb2s1=wb2.worksheets[0]
maxr=wb2s1.max_row    #最大行数
maxc=wb2s1.max_column  #最大列数


#db1=[]
db1=[["" for i in range(1+maxr)]for i in range(1+maxc)]
#val1=''
for c in range(1,maxc+1):
	for r in range(1,maxr+1):
#		val = wbs1.cell(row=r,column=c).value
		val = wb2s1.cell(row=r,column=c).value
		db1[c][r] = (val if val is not None else '')


#for c in range(7,8):
for c in range(7,maxc+1):
#for c in range(84,86+1):
    data=db1[c][53:1+92]
    data=[i for i in data if i != '']
#    mean=np.mean(data)
    db1[c].append(db1[c][48])
    if data!=[] :
        db1[c].append(np.mean(data))
        db1[c].append(np.std(data))
        db1[c].append(max(data))
        db1[c].append(min(data))


#wb2s2=wb2.create_sheet("ratio",1)
wb2s2=wb2.worksheets[1]
maxc = len(db1)
maxr = len(db1[10])
#for c in range(maxc):
for c in range(10):
#	for r in range(maxr):
	for r in range(maxr):
		wb2s2.cell(row=r+1,column=c+1).value=db1[c][r]

wb2.save('test2.xlsx')
#x=db1[2][53:1+92]
#x=max(x)
#x=min(x)
#x=np.mean(x)
#x=np.std(x)
#x=format(x,'0.2f')
#db1[2].append("xxxxxxx")
#db1[2].append({"mean":123})
#db1[2].append(db1[2][48])
#x=db1[2]
#x=len(db1)
x=len(db1[0])
#x=db1

print("dir x :\n ",dir(x))
print("x type is:\n ",type(x))
print("x is:\n ",x)


exit(0)

for c in range(1,1+maxc):
	for r in range(1,1+maxr):
		wbs2.cell(row=r,column=c).value=db1[c][r]

#x=str(db1[1][1])
x=db1[1][1]
#x=db1[1]
wbs2.cell(row=1,column=1).value=x
wb2.save('test2.xlsx')

print("dir x :\n ",dir(x))
print("x type is:\n ",type(x))
print("x is:\n ",x)
exit(0)

