
import numpy as np
#import pandas as pd
from pathlib import Path
import struct
import matplotlib.pyplot as plt

def pt(x,display=1,fexit=0):
    print("dir is:\n",dir(x))
    print("type is:\n ",type(x))
    if display==1: print("variable is:\n",x)
    if fexit!=0: exit(0)
    return x

mnist_path=Path('./mnist')
train_img_path=mnist_path/'train-images-idx3-ubyte'
train_lab_path=mnist_path/'train-labels-idx1-ubyte'
test_img_path=mnist_path/'t10k-images-idx3-ubyte'
test_lab_path=mnist_path/'t10k-labels-idx1-ubyte'

train_num=50000
valid_num=10000
test_num=10000

with open(train_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    tmp_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28,1)
    train_img=tmp_img[:train_num]
    valid_img=tmp_img[train_num:]
with open(test_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    test_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28*28,1)
with open(train_lab_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    tmp_lab=np.fromfile(f,dtype=np.uint8)
    train_lab=tmp_lab[:train_num]
    valid_lab=tmp_lab[train_num:]
with open(test_lab_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    test_lab=np.fromfile(f,dtype=np.uint8)
def plot(img,h=28,w=28):
    img=img.reshape(h,-1)
    plt.imshow(img,cmap='gray')
    plt.show()

#n=np.random.randint(50000)
#img=train_img[n].reshape(28,28)
#plt.imshow(img,cmap='gray')
#plt.show()
