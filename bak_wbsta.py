#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openpyxl
from matplotlib import pyplot as plt 

class wbsta:
    def __init__(self,wbfile):
        self.wbfile=wbfile
        self.db=[]
        self.nrowsPreData=5
        self.unitIndex=1

    def wb2db(self,wbfile):
        wb=openpyxl.load_workbook(wbfile)
        sheet1=wb.get_sheet_by_name(wb.sheetnames[0])
        dcols=sheet1.columns
        db=[]
        for dcol in dcols:
            dtmp=[i.value if i.value is not None else '' for i in dcol]
            db.append(dtmp)
        return db    

    def finddata(self,param):
        qd=0
        for i in self.db:
            if param==i[0]: q=[i[0:self.nrowsPreData-1][self.nrowsPreData:-1]]; qd=1
            if qd==1: break 
        return q

    #def sortdata(data):
        #return sorted(data)


    def find2data_sort(self,param1,param2,keyi=0):
        db=self.wb2db(wbfile)
        d1=[];d2=[];dn1=0;dn2=0;da=[];unit=''
        for i in db:
            if i[0]==param1: d1=i; dn1=1
            elif i[0]==param2: d2=i; dn2=1
            if dn1&dn2==1: unit=d1[self.unitIndex];break
        numdata=list(range(1,1+max(len(d1),len(d2))))
        dt1=d1[self.nrowsPreData:-1];dt2=d2[self.nrowsPreData:-1]
        da=list(zip(*sorted(zip(numdata,dt1,dt2),key=lambda x:x[1+keyi])))
        return da,unit
                    
    #def plot2(self,d1,d2,t1='',t2='',unit=''):
    def plot2(self,param1,param2,unit,data1,data2):
        x=list(range(1,1+len(data1)))
        plt.title('data1='+param1+'\ndata2='+param2) 
        plt.xlabel('number') 
        plt.ylabel(unit) 
        plt.plot(x,data1,x,data2)
        plt.show()


    def cmp2(self,param1,param2):
        da,unit = self.find2data_sort(param1,param2)
        self.plot2(param1,param2,unit,da[1],da[2])





wbfile='data2.xlsx'
q=wbsta(wbfile)
#q.wb2db(wbfile)

x=q
#x=q.db

param1='v1p0_bf'
param2='v1p0_af'

q.cmp2(param1,param2)


#print(dir(x))
print(type(x))
print(x)
