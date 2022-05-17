
import numpy as np
#import pandas as pd
from pathlib import Path
import struct
import matplotlib.pyplot as plt

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
#		tmp_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28,28)
		tmp_img=np.fromfile(f,dtype=np.uint8).reshape(-1,1,28,28)
		train_img=tmp_img[:train_num]
		valid_img=tmp_img[train_num:]
with open(test_img_path,'rb') as f:
		struct.unpack('>4i',f.read(16))
#		test_img=np.fromfile(f,dtype=np.uint8).reshape(-1,28,28)
		test_img=np.fromfile(f,dtype=np.uint8).reshape(-1,1,28,28)
with open(train_lab_path,'rb') as f:
		struct.unpack('>2i',f.read(8))
		tmp_lab=np.fromfile(f,dtype=np.uint8)
		train_lab=tmp_lab[:train_num].reshape(-1,1,1)
		valid_lab=tmp_lab[train_num:].reshape(-1,1,1)
with open(test_lab_path,'rb') as f:
		struct.unpack('>2i',f.read(8))
		test_lab=np.fromfile(f,dtype=np.uint8)
		test_lab=test_lab.reshape(-1,1,1)
#
#def plot(img):
#		 plt.imshow(img,cmap='gray')
#		 plt.show()
#n=np.random.randint(50000)
#img=train_img[n].reshape(28,28)
#plt.imshow(img,cmap='gray')
#plt.show()
