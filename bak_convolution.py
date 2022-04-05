import numpy as np
import mnist

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




#bb=np.zeros((4,3,9),dtype=int)
#for j in range(n):
#    for i in range(m):
##        bb[j,i,:]=aaa[i:i+k,j:j+k].reshape(-1)
        

## ==========
## convolution
## ==========
def conv(img,k=3,kernal=np.zeros((k,k))):
    if np.all(kernal==0):
        kernal[int(k/2),int(k/2)]=1
    m=img.shape[0]
    n=img.shape[1]
    imga=np.pad(img,int(k/2))
    img2rows=np.zeros((n,m,k**2))
    for col in range(n):
        for row in range(m):
            img2rows[col,row,:]=imga[row:row+k,col:col+k].reshape(-1)
    conv_out=(img2rows@(kernal.reshape(-1))).T
#    return conv_out


    print(kernal)
    print(img2rows)
    print(imga)
    print(conv_out)
    print(conv_out.shape)
#    print(kernal)

a=np.arange(m*n).reshape(m,n)
conv(a,3)

