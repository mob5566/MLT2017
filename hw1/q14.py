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

  b = m.rho[0]
  sv_coef = [sv_c[0] for sv_c in m.get_sv_coef()]
  sv = []
  for v in m.get_SV():
    t = []
    for i in range(1, 3):
      if i in v:
          t.append(v[i])
      else: t.append(0)
    sv.append(t)

  for i in range(len(sv_coef)):
    if abs(sv_coef[i]) != c:
      break
  dis = 0.0
  for x, ay in zip(sv, sv_coef):
    dis += ay * np.exp(-100 * ((x[0]-sv[i][0])**2 + (x[1]-sv[i][1])**2))
  dis = abs(dis + y[i]*b)
  D.append(dis)

# Plot the result

plt.figure()
plt.cla()

plt.plot(C, D, 'r*--')

plt.title('Q.14')
plt.xlabel('$\log_{10}C$')
plt.ylabel('distance to hyperplane')

plt.savefig('q14.png')
