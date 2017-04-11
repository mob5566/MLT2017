'''
# Machine Learning Techniques - Hw1 Q.14

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
from svmutil import *

if len(sys.argv)!=3:
  print('Usage: python q14.py <train_data> <test_data>')
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
C = [-3., -2., -1., 0., 1.]
D = []

for c in C:
  prob = svm_problem(y, X)
  param = svm_parameter('-s 0 -t 2 -c {} -h 0 -g 80'.format(10.**c))
  m = svm_train(prob, param)

  for sv, al in zip(m.get_SV(), m.get_sv_coef()):
    if not np.isclose(c, al):
      fsv = sv
      break

  dis = 0.
  for sv, al in zip(m.get_SV(), m.get_sv_coef()):
    dis += al[0]*np.exp(-80.*((sv[1]-fsv[1])**2+(sv[2]-fsv[2])**2))
  D.append(dis)

# Plot the result

plt.figure()
plt.cla()

plt.plot(C, D, 'r*--')

plt.title('Q.14')
plt.xlabel('$\log_{10}C$')
plt.ylabel('distance to hyperplane')

plt.savefig('q14.png')
