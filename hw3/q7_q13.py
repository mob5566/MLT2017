'''
# Machine Learning Techniques - Hw3 Q.7 to Q.13

## Experiments with Adaptive Boosting

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import sys
import os

import matplotlib.pyplot as plt

import hw3_model as md

if __name__ == '__main__':
  if len(sys.argv)!=3:
    print('Usage: python q7_q13.py <train_data> <test_data>')
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

  # Train AdaBoost
  maxiter = 300
  model = md.adaboost(maxiter)
  model.fit(X, y)

  # Run experiments
  print('Question 7.')

  t = np.arange(maxiter)
  
  plt.title('Question 7')
  plt.xlabel(r'$t$')
  plt.ylabel(r'$E_\mathsf{in}(g_t)$')

  plt.plot(t+1, [1-md.accuracy(y, model.G[i].predict(X)) for i in t])
  plt.savefig('Q7.png')
  plt.cla()

  print('Ein(g_1) = {:.3f}'.format(1-md.accuracy(y, model.G[0].predict(X))))
  print('alpha_1 = {:.3f}'.format(model.alpha[0]))

  print('\nQuestion 9.')

  GEins = []
  ypred = np.zeros(N)

  for alpha, g in zip(model.alpha, model.G):
    ypred += alpha * g.predict(X)
    GEins.append(1-md.accuracy(y, (ypred>=0).astype(float)*2-1))

  plt.title('Question 9')
  plt.xlabel(r'$t$')
  plt.ylabel(r'$E_\mathsf{in}(G_t)$')

  plt.plot(t+1, GEins)
  plt.savefig('Q9.png')
  plt.cla()

  print('Ein(G) = {:.3f}'.format(1.0-md.accuracy(y, model.predict(X))))
