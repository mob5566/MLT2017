'''
# Machine Learning Techniques - Hw1 Q.2

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def ker(u, v):
  return (2.0+np.dot(u, v))**2

# Set data
x = []

x.append([ 1.,  0.])
x.append([ 0.,  1.])
x.append([ 0., -1.])
x.append([-1.,  0.])
x.append([ 0.,  2.])
x.append([ 0., -2.])
x.append([-2.,  0.])

x = np.array(x)

y = []

y.append(-1.)
y.append(-1.)
y.append(-1.)
y.append( 1.)
y.append( 1.)
y.append( 1.)
y.append( 1.)

y = np.array(y)
n = x.shape[0]

Q = matrix([[y[i]*y[j]*ker(x[i], x[j]) for i in xrange(n)] for j in xrange(n)], tc='d')
p = matrix([-1.0]*n, tc='d')

G = matrix(-np.eye(n), tc='d')
h = matrix([0.0]*n, tc='d')

A = matrix(y, tc='d').T
c = matrix(0.0, tc='d')

# Solve quadratic programming
sol = solvers.qp(Q, p, G, h, A, c)

alpha = np.array(sol['x']).reshape(-1)

print('alpha:')
print(alpha)

svi = np.array([i for i in xrange(n) if not np.isclose(alpha[i], 0)])
print('support vector index: ', svi+1)

svx = x[svi[0]]
svy = y[svi[0]]
b = svy - np.sum([alpha[i]*y[i]*ker(x[i], svx) for i in svi])
print('b = ', b)
