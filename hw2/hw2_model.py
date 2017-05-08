'''
# Machine Learning Techniques - Hw2 Model

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw

* Implement learning models used in homework 2

  * Least-Squares SVM
    * Gaussian-RBF kernel
    * Linear kernel

  * Bagging by uniform aggregation

* 0/1 error
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

'''
## Least-Squares SVM
'''
class LSSVM(object):
  # Constructor
  def __init__(self, kernel, ld=0.001):
    self.ld = ld
    self.kernel = kernel

  # Training
  def fit(self, X, y):

    assert(type(X)==np.ndarray and len(X.shape)==2)
    assert(type(y)==np.ndarray and len(y.shape)==2)

    K = np.array([[self.kernel(x1, x2) for x1 in X] for x2 in X])

    self.beta = np.dot(np.linalg.inv(self.ld*np.eye(X.shape[0])+K), y)
    self.X = X.copy()

  # Prediction
  def predict(self, X):

    assert(type(X)==np.ndarray and len(X.shape)==2)
    assert(self.X.shape[1] == X.shape[1])
    
    Z = np.array([[self.kernel(x1, x2) for x2 in X] for x1 in self.X])

    return self.beta.T.dot(Z)

  # Static methods
  @staticmethod
  def rbf_kernel(gamma=0.125):
    return lambda x, xp: np.exp(-gamma*np.linalg.norm(x-xp)**2)

  @staticmethod
  def linear_kernel():
    return lambda x, xp: np.dot(x, xp)

class Bagging(object):
  def __init__(self, baseModels):
    assert(type(baseModels)==list)

    self.baseModels = baseModels

  # Training 
  def fit(self, X, y):

    assert(type(X)==np.ndarray and len(X.shape)==2)
    assert(type(y)==np.ndarray and len(y.shape)==2)
    
    for md in self.baseModels:
      trainmask = np.random.randint(0, X.shape[0], X.shape[0])

      md.fit(X[trainmask], y[trainmask])
  
  # Prediction
  def predict(self, X):

    assert(type(X)==np.ndarray and len(X.shape)==2)

    ret = np.zeros(X.shape[0]).reshape(1, -1)
    
    for md in self.baseModels:
      ret += np.sign(md.predict(X))
    
    ret[np.isclose(ret, 0)] = 1

    return ret

'''
## 1/0 error
'''
def mismatch(y1, y2):
  y1 = y1.reshape(-1)
  y2 = y2.reshape(-1)
  assert(y1.shape[0]==y2.shape[0])
  return 1.0-np.isclose(np.sign(y1), np.sign(y2)).mean()
