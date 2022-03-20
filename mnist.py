
import pandas as pd
from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt

def pt(x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x

mnist_path=Path('./mnist')

#pt(mnist_path)

train_img_path=mnist_path/'train-images-idx3-ubyte'
train_lab_path=mnist_path/'train-labels-idx1-ubyte'
test_img_path=mnist_path/'t10k-images-idx3-ubyte'
test_lab_path=mnist_path/'t10k-labels-idx1-ubyte'
#pt(train_img_path)

#f=open(train_img_path,'rb')
with open(train_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    a=np.fromfile(f,dtype=np.uint8)
    train_img=a.reshape(-1,28*28)
#pt(a)
#pt(a.ndim)
#pt(a.shape)

n=np.random.randint(50000)
img=train_img[n].reshape(28,28)
#pt(img)
plt.imshow(img,cmap='gray')
plt.show()
