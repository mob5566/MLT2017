'''
# Machine Learning Techniques - Hw1 Q.16

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
from svmutil import *

if len(sys.argv)!=3:
  print('Usage: python q16.py <train_data> <test_data>')
  sys.exit(-1)

trainf = sys.argv[1]
testf = sys.argv[2]

# Setup data
X = []
y = []
with open(trainf, 'rb') as f:
  for line in f.readlines():
    line = line.split()
    X.append({1:float(line[1]), 2:float(line[2])})
    y.append(1 if np.isclose(0.0, float(line[0])) else -1)

X = np.array(X)
y = np.array(y)

# Train SVM
gamma = [-1., 0., 1., 2., 3.]
sel = [0.]*5
rdnp = np.arange(len(X))

for i in xrange(100):
  Eval = []
  np.random.shuffle(rdnp)
  Xtrain = X[rdnp[1000:]]
  ytrain = y[rdnp[1000:]]
  Xval = X[rdnp[:1000]]
  yval = y[rdnp[:1000]]
 
  for g in gamma:
    prob = svm_problem(ytrain, Xtrain)
    param = svm_parameter('-s 0 -t 2 -c 0.1 -h 0 -g {}'.format(10.**g))
    m = svm_train(prob, param)
    labs, acc, _ = svm_predict(yval, Xval, m)
    Eval.append(100.-acc[0])
  sel[np.argmin(Eval)] += 1

# Plot the result

plt.figure()
plt.cla()

plt.bar(gamma, sel)

plt.title('Q.16')
plt.xlabel('$\log_{10}\gamma$')
plt.ylabel('number of time be selected')

plt.savefig('q16.png')
