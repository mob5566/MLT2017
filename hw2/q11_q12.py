'''
# Machine Learning Techniques - Hw2 Q.11 and Q.12

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
    print('Usage: python q11_q12.py <train_data>')
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
    y = data[:,  -1].reshape(-1, 1)

  trainX = X[:400]
  testX  = X[400:]
  trainy = y[:400]
  testy  = y[400:]

  # Run experiment
  gamma = [32, 2, 0.125]
  lamb = [0.001, 1.0, 1000]
  einGrid = []
  eoutGrid = []

  for g in gamma:
    eins = []
    eouts = []

    for l in lamb:
      model = md.LSSVM(md.LSSVM.rbf_kernel(g), l)

      model.fit(trainX, trainy) 

      eins.append(md.mismatch(model.predict(trainX), trainy))
      eouts.append(md.mismatch(model.predict(testX), testy))

    einGrid.append(eins)
    eoutGrid.append(eouts)

  # Show result
  print('Question 11.')
  print('Ein Grid (row gamma/col lambda)')
  print(np.array(einGrid))
  print('')
  print('Question 12.')
  print('Eout Grid (row gamma/col lambda)')
  print(np.array(eoutGrid))
  print('')
