import numpy as np

k=3
m=3
n=4
kk=np.zeros((k,k),dtype=int)
kk[1,1]=1
kk=kk.reshape(-1)
aa=np.arange(m*n,dtype=int).reshape(m,n)
aaa=np.pad(aa,1)
aaa
# kk




bb=np.zeros((4,3,9),dtype=int)
for j in range(n):
    for i in range(m):
        bb[j,i,:]=aaa[i:i+k,j:j+k].reshape(-1)
        






