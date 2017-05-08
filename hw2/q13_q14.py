'''
# Machine Learning Techniques - Hw2 Q.13 and Q.14

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import sys
import os

from sklearn.svm import SVR

import hw2_model as md

if __name__ == '__main__':
  if len(sys.argv)!=2:
    print('Usage: python q13_q14.py <train_data>')
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
    y = data[:,  -1]

  trainX = X[:400]
  testX  = X[400:]
  trainy = y[:400]
  testy  = y[400:]

  # Run experiment
  gammas = [32, 2, 0.125]
  CC = [0.001, 1.0, 1000]
  eps = 0.5
  einGrid = []
  eoutGrid = []

  for g in gammas:
    eins = []
    eouts = []

    for c in CC:
      model = SVR(gamma=g, C=c, epsilon=eps, kernel='rbf')

      model.fit(trainX, trainy) 

      eins.append(md.mismatch(model.predict(trainX), trainy))
      eouts.append(md.mismatch(model.predict(testX), testy))

    einGrid.append(eins)
    eoutGrid.append(eouts)

  # Show result
  print('Question 13.')
  print('Ein Grid (row gamma/col C)')
  print(np.array(einGrid))
  print('')
  print('Question 14.')
  print('Eout Grid (row gamma/col C)')
  print(np.array(eoutGrid))
  print('')
