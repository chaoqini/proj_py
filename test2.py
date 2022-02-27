#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openpyxl
from matplotlib import pyplot as plt 
import wbsta
#import numpy as np
##x = np.arange(1,4)



class wbsta:
    def __init__(self,wb):
        self.wb=wb
        self.shNames=wb.sheetnames
        pass

    def wb2db(self,wb):
        wb=openpyxl.load_workbook(wb)
        ws1=wb.get_sheet_by_name(self.shNames[0])
        dcols=ws1.columns
        db=[]
        for col in dcols:
            ctmp=[i.value if i.value is not None else '' for i in col]
            db.append()

wb='data2.xlsx'
x=wbsta.shNames(wb)
print(x)


    #x = np.arange(1,4) 
    #print( x )
#
#y = [ i if i==3 else '' for i in x] 
#print(y)
#
#db=range(10)
#
#n=0
#dn1=0
#dn2=0
#for i in db:                                     
#    if i==2:                                     
#        y1=i                                   
#        dn1=1                                 
#    elif i==6:
#        y2=i                            
#        dn2=1 
#    n=n+1        
#    #break if dn1&dn2 == 1 else pass
#    if dn1&dn2 == 1 : break
#
#
##d1=list(range(9,1))
#d1=list(range(9,0,-1))
#d2=list(range(11,20))
#
#print(type(d1))
#
#d3=zip(d1,d2)
#d4=list(d3)
#
#d5=sorted(d3)
#d6=sorted(d4)
#
#d7=sorted(d6,key=lambda x:x[1])
#d8=sorted(d4,key=lambda x:x[0])
#
#d9=list(zip(*d8))
#
#d10=d9[0]
#d11=d9[1]
#
##plt.plot(d10,d11)
##plt.show()
#
#d12=list(d10)
#d13=list(d11)
#
#d12=[d12[0:2],d12[3:-1]]
#d13=[d13[0:2],d13[3:-1]]
#
#
#def sortr(d1,d2,i=0):
#    da=list(zip(*sorted(zip(d1,d2),key=lambda x:x[i])))
#    return da
#
#
#d14=sortr(d1,d2)
#d15=sortr(d1,d2,1)
#
#
#d16=sortr(d1,d2).[0]
#
#print(d16)
#

