'''
# Machine Learning Techniques - Hw1 Q.11

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
from svmutil import *

if len(sys.argv)!=3:
  print('Usage: python q11.py <train_data> <test_data>')
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
C = [-5., -3., -1., 1., 3.]
W = []

for c in C:
  prob = svm_problem(y, X)
  param = svm_parameter('-s 0 -t 0 -c {} -h 0'.format(10.**c))
  m = svm_train(prob, param)
  w = [0.0]*2
  for sv, al in zip(m.get_SV(), m.get_sv_coef()):
    w[0] += sv[1]*al[0]
    w[1] += sv[2]*al[0]
  W.append(np.sqrt(w[0]**2+w[1]**2))

# Plot the result

plt.figure()
plt.cla()

plt.plot(C, W, 'r*--')

plt.title('Q.11')
plt.xlabel('$\log_{10}C$')
plt.ylabel('$\Vert w\Vert$')

plt.savefig('q11.png')
