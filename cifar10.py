from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


cifar10_path=Path('./cifar-10')
#print(cifar10_path)
databatch1_path=cifar10_path/'data_batch_1'
databatch2_path=cifar10_path/'data_batch_2'
databatch3_path=cifar10_path/'data_batch_3'
databatch4_path=cifar10_path/'data_batch_4'
databatch5_path=cifar10_path/'data_batch_5'
testbatch_path=cifar10_path/'test_batch'
#print(train_img_path)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
#        dict = pickle.load(fo, encoding='bytes')
        dict = pickle.load(fo, encoding='latin1')
    return dict
##[b'batch_label', b'labels', b'data', b'filenames']

#aa=unpickle(testbatch_path)
databatch=unpickle(databatch2_path)
#x=databatch['data'].reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
x=databatch['data'].reshape(10000,3,32,32).transpose(0,2,3,1)
x=x[20]
#x=databatch
#x=list(databatch.keys())
#x=aa[b'batch_label']
#x=aa[b'labels']
#x=aa[b'filenames']
#x=aa[b'filenames'][0]
#x=np.array(aa[b'data'])
#x=x[1].reshape(32,32,-1)
#x=x[:,:,0]
print(x)
print(x.shape)
print(type(x))
plt.imshow(x)
plt.show()

#astype('uint8'
