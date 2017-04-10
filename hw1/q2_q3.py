'''
# Machine Learning Techniques - Hw1 Q.2 & Q.3

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

#
# Question 2
#

Q = matrix([[y[i]*y[j]*ker(x[i], x[j]) for i in xrange(n)] for j in xrange(n)], tc='d')
p = matrix([-1.0]*n, tc='d')

G = matrix(-np.eye(n), tc='d')
h = matrix([0.0]*n, tc='d')

A = matrix(y, tc='d').T
c = matrix(0.0, tc='d')

# Solve quadratic programming
sol = solvers.qp(Q, p, G, h, A, c)

alpha = np.array(sol['x']).reshape(-1)

print('Hw1 Q.2\n')

print('alpha:')
print(alpha)

svi = np.array([i for i in xrange(n) if not np.isclose(alpha[i], 0)])
print('support vector index: ', svi+1)

svx = x[svi[0]]
svy = y[svi[0]]
b = svy - np.sum([alpha[i]*y[i]*ker(x[i], svx) for i in svi])
print('b = ', b)

print('')

#
# Question 3
#

# Hyper-plane
x2 = np.linspace(-2.4, 2.4, 100)
x1 = np.array([np.roots([0.533, -2.132, -1.662+0.668*v*v]).real for v in x2])

# Plot data
plt.figure(1, figsize=(7.5, 7.5))

plt.plot(x[y==1, 0], x[y==1, 1], 'ro', markersize=5, label='positive')
plt.plot(x[y==-1, 0], x[y==-1, 1], 'bx', markersize=5, label='negative')

# Plot seperating hyperplane
plt.plot(x1[:, 0], x2, 'g', linewidth=2.5, label='hyperplane')
plt.plot(x1[:, 1], x2, 'g', linewidth=2.5)

plt.xlim([-2.5, 5])
plt.ylim([-2.5, 2.5])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'$\mathcal{X}\ space$')
plt.legend(loc='lower right')

plt.savefig('q3.png')
