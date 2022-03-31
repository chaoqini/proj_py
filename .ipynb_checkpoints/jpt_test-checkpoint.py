{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "36fe7440",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The accuracy is : 90.64%\n",
      "Label number is : 9\n",
      "Precdict number is : 9\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import ann\n",
    "import numpy as np\n",
    "n=np.random.randint(10000)\n",
    "ann.show(n)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.10.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
