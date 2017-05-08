'''
# Machine Learning Techniques - Hw2 Q.15 and Q.16

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import sys
import os

import hw2_model as md

if __name__ == '__main__':
  if len(sys.argv)!=2:
    print('Usage: python q15_q16.py <train_data>')
    sys.exit(-1)

  trainf = sys.argv[1]

  if not os.path.exists(trainf):
    print("{} doesn't exist!".format(trainf))
    sys.exit(-1)

  # Setup data
  X = []
  y = []

  with open(trainf, 'rb') as f:
    data = np.array([line.split() for line in f.readlines()]).astype(float)

    X = data[:, :-1]
    X = np.append(X, np.ones(X.shape[0]).reshape(-1, 1), axis=1)
    y = data[:,  -1].reshape(-1, 1)

  trainX = X[:400]
  testX  = X[400:]
  trainy = y[:400]
  testy  = y[400:]

  # Run experiment
  lamb = [0.01, 0.1, 1.0, 10.0, 100.0]
  numModel = 300

  eins = []
  eouts = []

  for l in lamb:
    basemd = [md.LSSVM(md.LSSVM.linear_kernel(), l) for i in xrange(numModel)]
    model = md.Bagging(basemd)

    model.fit(trainX, trainy) 

    eins.append(md.mismatch(model.predict(trainX), trainy))
    eouts.append(md.mismatch(model.predict(testX), testy))

  # Show result
  print('Question 15.')
  print('Ein Grid')
  print(np.array(eins))
  print('')
  print('Question 16.')
  print('Eout Grid')
  print(np.array(eouts))
  print('')
