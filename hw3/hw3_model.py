'''
# Machine Learning Techniques - Hw3 Model

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw

* Implement learning models used in homework 3

  * Decision Stump
  * Adaptive Boosting

'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

eps = 1e-8

'''
## Decision Stump
'''
class decision_stump(object):
  # Constructor
  def __init__(self):
    self.theta = 0
    self.i = 0
    self.s = 1.0
    self.d = 0

  # Training
  def fit(self, X, y, weighted=None, sortedIdx=None):

    assert(type(X)==np.ndarray and len(X.shape)==2)
    assert(type(y)==np.ndarray and len(y.shape)==1)
    assert(len(X) == len(y))

    N, d = X.shape
    self.d = d

    if weighted is None:
      weighted = np.ones(N)/N

    # Get sorted index of each dimension
    if sortedIdx is None:
      sortedIdx = []

      for i in xrange(d):
        idx = range(N)
        idx.sort(key=lambda x: X[x, i])
        sortedIdx.append(idx)

      sortedIdx = np.array(sortedIdx).T

    # Training
    minein = 1.0

    for i in xrange(d):
      last = -np.inf

      spliter = X[sortedIdx[:, i], i]

      for v in spliter:
        mid = (last+v)*0.5
        ypred = 2*(X[:, i]>=mid).astype(float)-1
        ein = 1.0 - accuracy(y, ypred, weighted)

        if ein < minein or (1-ein) < minein:
          minein = min(ein, 1-ein)
          self.theta = mid
          self.s = 1.0 if ein < 0.5 else -1.0
          self.i = i

        last = v

    return sortedIdx

  # Prediction
  def predict(self, X):
    
    assert(type(X)==np.ndarray and len(X.shape)==2)

    N, d = X.shape

    assert(self.d == d)

    return ((X[:, self.i]>=self.theta).astype(float)*2-1)*self.s


'''
## Adaptive Boosting
'''
class adaboost(object):
  # Constructor
  def __init__(self, maxiter=300):
    self.maxiter = maxiter

  # Training
  def fit(self, X, y, weighted=None):

    assert(type(X)==np.ndarray and len(X.shape)==2)
    assert(type(y)==np.ndarray and len(y.shape)==1)
    assert(len(X) == len(y))

    N, d = X.shape
    self.d = d

    if weighted is None:
      weighted = np.ones(N)/N

    self.G = []
    self.alpha = []
    self.u = []

    # Training
    idx = None
    for i in xrange(self.maxiter):
      g = decision_stump()
      idx = g.fit(X, y, weighted, idx)

      # Update
      ypred = g.predict(X)
      e = 1 - accuracy(y, ypred, weighted)
      alpha = 0.5*np.log((1-e)/e)
      weighted = weighted * np.exp(alpha*(1-2*np.isclose(y, ypred).astype(float)))

      self.G.append(g)
      self.alpha.append(alpha)
      self.u.append(weighted)

  # Prediction
  def predict(self, X):
    
    assert(type(X)==np.ndarray and len(X.shape)==2)

    N, d = X.shape

    assert(self.d == d)

    ypred = np.zeros(N)

    for alpha, g in zip(self.alpha, self.G):
      ypred += alpha*g.predict(X)

    return ((ypred>=0).astype(float)*2-1)

"""
## Metrics
"""

def accuracy(ytrue, ypred, weighted=None):
  assert(type(ytrue)==np.ndarray and len(ytrue.shape)==1)
  assert(type(ypred)==np.ndarray and len(ypred.shape)==1)
  assert(len(ytrue)==len(ypred))

  N = len(ypred)

  if weighted is None:
    weighted = np.ones(N)/N

  acc = np.isclose(ytrue, ypred).astype(float).dot(weighted)/(weighted.sum()+eps)

  return acc
