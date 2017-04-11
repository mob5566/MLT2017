'''
# Machine Learning Techniques - Hw1 Q.12 and Q.13

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
from svmutil import *

if len(sys.argv)!=3:
  print('Usage: python q12_q13.py <train_data> <test_data>')
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
    y.append(1 if np.isclose(8.0, float(line[0])) else -1)

testX = []
testy = []
with open(testf, 'rb') as f:
  for line in f.readlines():
    line = line.split()
    testX.append({1:float(line[1]), 2:float(line[2])})
    testy.append(1 if np.isclose(8.0, float(line[0])) else -1)

# Train SVM
C = [-5., -3., -1., 1., 3.]
ein = []
nsv = []

for c in C:
  prob = svm_problem(y, X)
  param = svm_parameter('-s 0 -t 1 -c {} -h 0 -g 1 -r 1 -d 2'.format(10.**c))
  m = svm_train(prob, param)
  labs, acc, _ = svm_predict(y, X, m)
  ein.append(100.-acc[0])
  nsv.append(m.get_nr_sv())

# Plot the result
plt.figure()
plt.cla()

plt.plot(C, ein, 'r*--')

plt.title('Q.12')
plt.xlabel('$\log_{10}C$')
plt.ylabel('$E_{in}$')

plt.savefig('q12.png')

plt.figure()
plt.cla()

plt.plot(C, nsv, 'r*--')

plt.title('Q.13')
plt.xlabel('$\log_{10}C$')
plt.ylabel('number of support vectors')

plt.savefig('q13.png')
