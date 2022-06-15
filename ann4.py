{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a55fa101",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init_params: ch= [1, 2, 3, 4, 10]\n",
      "init_params: ch= [1, 2, 3, 4, 10]\n",
      "init_params: ch= [1, 4, 8, 12, 10]\n",
      "heyperparams_test: ...\n",
      "heyperparams_test: layers = 4\n",
      "heyperparams_test: learning rate lr0 = 0.002\n",
      "heyperparams_test: learning rate klr = 0.9995\n",
      "heyperparams_test: batch = 32\n",
      "heyperparams_test: isupdate = 0\n",
      "params k0.shape= (4, 1, 3, 3)\n",
      "params gama0.shape= (4, 1, 1)\n",
      "params beta0.shape= (4, 1, 1)\n",
      "params k1.shape= (8, 4, 3, 3)\n",
      "params gama1.shape= (8, 1, 1)\n",
      "params beta1.shape= (8, 1, 1)\n",
      "params k2.shape= (12, 8, 3, 3)\n",
      "params gama2.shape= (12, 1, 1)\n",
      "params beta2.shape= (12, 1, 1)\n",
      "params w3.shape= (10, 12, 28, 28)\n",
      "active function g[0] is: Relu\n",
      "active function g[1] is: Relu\n",
      "active function g[2] is: Relu\n",
      "active function g[3] is: softmax\n",
      "active function g[4] is: cross_entropy\n",
      "============================================================\n",
      "hyperparams_test runing iteration = 1/8\n",
      "hyperparams_test runing: lri = 2.000e-03\n",
      "Hyperparams_test: L2_normalize_k0 = 1.75e-01\n",
      "Hyperparams_test: L2_normalize_gama0 = 5.00e-01\n",
      "Hyperparams_test: L2_normalize_beta0 = 0.00e+00\n",
      "Hyperparams_test: L2_normalize_k1 = 5.56e-02\n",
      "Hyperparams_test: L2_normalize_gama1 = 3.54e-01\n",
      "Hyperparams_test: L2_normalize_beta1 = 0.00e+00\n",
      "Hyperparams_test: L2_normalize_k2 = 3.39e-02\n",
      "Hyperparams_test: L2_normalize_gama2 = 2.89e-01\n",
      "Hyperparams_test: L2_normalize_beta2 = 0.00e+00\n",
      "Hyperparams_test: L2_normalize_w3 = 3.26e-03\n",
      "Train input X.shape=(1562, 32, 1, 28, 28), LAB.shape=(1562, 32, 1, 1)\n",
      "Training bath running ...\n",
      "Training iteration number = 0/1562\n"
     ]
    }
   ],
   "source": [
    "import ann\n",
    "import numpy as np\n",
    "import pickle\n",
    "import imp\n",
    "imp.reload(ann)\n",
    "(params,params_init,g)=ann.init_params(lays=4,imchin=4,dimch=4)\n",
    "params=ann.hyperparams_test(params,params_init,g,nloop=8,batch=32,isl2grad=1)\n",
    "with open('ann_p1.pkl', 'wb') as f: pickle.dump(params,f)\n",
    "with open('ann_p1.pkl', 'rb') as f: params2=pickle.load(f)\n",
    "for k,v in params2.items(): print('%s.shape='%k,v.shape)\n",
    "ann.show(params,g)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
