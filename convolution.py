import numpy as np
#import mnist


## ==========
## convolution
## ==========
def conv(img,k=3,kernal=np.zeros(())):
    kernal=np.zeros((k,k))
    if np.all(kernal==0):
        kernal[int(k/2),int(k/2)]=1
    m,n=img.shape
#    m=img.shape[0]
#    n=img.shape[1]
    imga=np.pad(img,int(k/2))
    img2rows=np.zeros((n,m,k**2))
    for col in range(n):
        for row in range(m):
            img2rows[col,row,:]=imga[row:row+k,col:col+k].reshape(-1)
    conv_out=(img2rows@(kernal.reshape(-1))).T
    return conv_out
#    print(kernal)
#    print(kernal.reshape(-1))
#    print(img2rows)
#    print(imga)
#    print(conv_out)
#    print(conv_out.shape)
#m=4
#n=5
#k=3
#ra0=np.array([1,0,1])
#ra1=np.array([1,0,1])
#ra2=np.array([1,0,1])
#ka=np.array([ra0,ra1,ra2])
#ka=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
#aa=np.arange(m*n).reshape(m,n)
#ca=conv(aa,k,ka)
#print(ca)
#print(ka)
#
