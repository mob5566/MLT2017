'''
# Machine Learning Techniques - Hw1 Q.15

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
from svmutil import *

if len(sys.argv)!=3:
  print('Usage: python q15.py <train_data> <test_data>')
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

testX = []
testy = []
with open(testf, 'rb') as f:
  for line in f.readlines():
    line = line.split()
    testX.append({1:float(line[1]), 2:float(line[2])})
    testy.append(1 if np.isclose(0.0, float(line[0])) else -1)

# Train SVM
gamma = [0., 1., 2., 3., 4.]
eout = []

for g in gamma:
  prob = svm_problem(y, X)
  param = svm_parameter('-s 0 -t 2 -c 0.1 -h 0 -g {}'.format(10.**g))
  m = svm_train(prob, param)
  labs, acc, _ = svm_predict(testy, testX, m)
  eout.append(100.-acc[0])

# Plot the result

plt.figure()
plt.cla()

plt.plot(gamma, eout, 'r*--')

plt.title('Q.15')
plt.xlabel('$\log_{10}\gamma$')
plt.ylabel('$E_{out}$')

plt.savefig('q15.png')
