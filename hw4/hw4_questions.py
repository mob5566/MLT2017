'''
# Machine Learning Techniques - Hw4 Q.12 to Q.16

## Experiments with Random Forest

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import sys
import os

import matplotlib.pyplot as plt

import hw4_model as md

if __name__ == '__main__':
  if len(sys.argv)!=3:
    print('Usage: python2 hw4_questions.py <train_data> <test_data>')
    sys.exit(-1)

  trainf = sys.argv[1]
  testf = sys.argv[2]

  if not os.path.exists(trainf) or not os.path.exists(testf):
    print("{} or {} do not exist!".format(trainf, testf))
    sys.exit(-1)

  # Setup data
  trainData = np.loadtxt(trainf)
  testData = np.loadtxt(testf)

  X = trainData[:, :-1]
  y = trainData[:,  -1]
  testX = testData[:, :-1]
  testy = testData[:,  -1]

  N, d = X.shape

  # Experiments
  ntrees = 30000

  rf = md.RandomForest(ntrees)
  rf.fit(X, y)

  print('\nQuestion 12\n')
  
  eins = []

  for tree in rf.G:
    eins.append(1-md.accuracy(y, tree.predict(X)))

  plt.figure(1)
  plt.title(r'Histogram of $E_\mathsf{in}(g_t)$')
  plt.xlabel(r'$E_\mathsf{in}$')
  plt.hist(eins, 1000)

  plt.savefig('Q12.png')
  print('Plot histogram')

  eins = []
  eouts = []
  predin = np.zeros(N)
  predout = np.zeros(len(testX))
  cnt = 0.0

  for tree in rf.G:
    predin += tree.predict(X)
    predout += tree.predict(testX)
    cnt += 1

    eins.append(1-md.accuracy(y, ((predin/cnt)>=0).astype(float)*2-1))
    eouts.append(1-md.accuracy(testy, ((predout/cnt)>=0).astype(float)*2-1))

  print('\nQuestion 13 and 14\n')

  plt.figure(2)
  plt.title(r'Question 13 and 14')
  plt.xlabel(r'$t$')
  plt.ylabel(r'$E(G_t)$')
  plt.plot(np.arange(1, ntrees+1), eins, label=r'$E_\mathsf{in}$')
  plt.plot(np.arange(1, ntrees+1), eouts, label=r'$E_\mathsf{out}$')
  plt.legend()

  plt.savefig('Q13_14.png')
  print('Plot the curve of Ein, Eout vs t')

  ######

  rf = md.RandomForest(ntrees, 1)
  rf.fit(X, y)

  eins = []
  eouts = []
  predin = np.zeros(N)
  predout = np.zeros(len(testX))
  cnt = 0.0

  for tree in rf.G:
    predin += tree.predict(X)
    predout += tree.predict(testX)
    cnt += 1

    eins.append(1-md.accuracy(y, ((predin/cnt)>=0).astype(float)*2-1))
    eouts.append(1-md.accuracy(testy, ((predout/cnt)>=0).astype(float)*2-1))

  print('\nQuestion 15 and 16\n')

  plt.figure(3)
  plt.title(r'Question 15 and 16')
  plt.xlabel(r'$t$')
  plt.ylabel(r'$E(G_t)$')
  plt.plot(np.arange(1, ntrees+1), eins, label=r'$E_\mathsf{in}$')
  plt.plot(np.arange(1, ntrees+1), eouts, label=r'$E_\mathsf{out}$')
  plt.legend()

  plt.savefig('Q15_16.png')
  print('Plot the curve of Ein, Eout vs t')
