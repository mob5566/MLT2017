'''
# Machine Learning Techniques - Hw4 Model

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw

* Implement learning models used in homework 3

  * Decision Tree
  * Random Froest

'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

eps = 1e-8

'''
## Decision Tree
'''
class dtree(object):
  def __init__(self, depth=0, maxdep=None):
    self.isLeaf = False
    self.val = None
    self.split_feat = None
    self.split_val = None
    self.childs = None
    self.maxdep = maxdep
    self.depth = depth
  
  def fit(self, X, y):

    N, d = X.shape

    # terminate 
    if np.all(np.isclose(X-X.mean(axis=0), 0)) or \
       np.all(np.isclose(y-y.mean(), 0)) or\
       (self.maxdep is not None and self.depth >= self.maxdep):

      self.isLeaf = True
      self.val = (y.sum()>=0).astype(float)*2-1
      return self

    minImp = np.inf

    for i in np.arange(d):
      interv = np.unique(X[:, i]) 

      for j in xrange(len(interv)-1):
        midv = (interv[j]+interv[j+1])*0.5
        smask = X[:, i] < midv
        gmask = np.logical_not(smask)

        imp = impurity(y[smask])*smask.sum() +\
              impurity(y[gmask])*gmask.sum()

        if imp < minImp:
          minImp = imp
          self.split_feat = i
          self.split_val = midv
          minSmask = smask
          minGmask = gmask
    
    self.childs = [dtree(self.depth+1, self.maxdep),
                   dtree(self.depth+1, self.maxdep)]

    self.childs[0].fit(X[minSmask], y[minSmask])
    self.childs[1].fit(X[minGmask], y[minGmask])

    return self

  def predict(self, X):
    if self.isLeaf:
      return self.val
    else:
      smask = X[:, self.split_feat]<self.split_val
      gmask = np.logical_not(smask)
      y = np.zeros(len(X))
      y[smask] = self.childs[0].predict(X[smask])
      y[gmask] = self.childs[1].predict(X[gmask])
    return y

def impurity(y):
  return 1-(np.isclose(1, y).mean())**2-(np.isclose(-1, y).mean())**2

'''
## Random Forest
'''
class RandomForest(object):
  # Constructor
  def __init__(self, maxiter=30000, maxdep=None):
    self.maxiter = maxiter
    self.maxdep = maxdep

  # Training
  def fit(self, X, y):

    assert(type(X)==np.ndarray and len(X.shape)==2)
    assert(type(y)==np.ndarray and len(y.shape)==1)
    assert(len(X) == len(y))

    N, d = X.shape
    self.d = d

    self.G = []

    # Training
    for i in xrange(self.maxiter):
      g = dtree(maxdep=self.maxdep)
      mask = np.random.randint(0, N, N)
      g.fit(X[mask], y[mask])
      self.G.append(g)

  # Prediction
  def predict(self, X):
    
    assert(type(X)==np.ndarray and len(X.shape)==2)

    N, d = X.shape

    assert(self.d == d)

    ypred = np.zeros(N)

    for g in self.G:
      ypred += g.predict(X)

    return ((ypred.mean()>=0).astype(float)*2-1)

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
