'''
# Machine Learning Techniques - Hw1 Q.1

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

import numpy as np
import matplotlib.pyplot as plt

# Set data
x = []
y = []

x.append([ 1,  0])
x.append([ 0,  1])
x.append([ 0, -1])
x.append([-1,  0])
x.append([ 0,  2])
x.append([ 0, -2])
x.append([-2,  0])

x = np.array(x)

y = []

y.append(-1)
y.append(-1)
y.append(-1)
y.append( 1)
y.append( 1)
y.append( 1)
y.append( 1)

y = np.array(y)

# Non-linear transform
px = 2*x[:, 1]**2 -4*x[:, 0] +1
px = px.reshape(-1, 1)
px = np.append(px, (x[:, 0]**2 -2*x[:, 1] -3).reshape(-1, 1), axis=1)

# Hyper-plane
x2 = np.linspace(-2.5, 2.5, 100)
x1 = (2*x2**2-3)/4.

# Plot data
plt.figure(1, figsize=(15, 7.5))

plt.subplot(121)
plt.plot(x[y==1, 0], x[y==1, 1], 'ro', markersize=5, label='positive')
plt.plot(x[y==-1, 0], x[y==-1, 1], 'bx', markersize=5, label='negative')

# Plot seperating hyperplane
plt.plot(x1, x2, 'g', linewidth=2.5, label='hyperplane')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'$\mathcal{X}\ space$')
plt.legend(loc='lower right')

plt.subplot(122)
plt.plot(px[y==1, 0], px[y==1, 1], 'ro', markersize=5, label='positive')
plt.plot(px[y==-1, 0], px[y==-1, 1], 'bx', markersize=5, label='negative')

# Plot seperating hyperplane
plt.plot([4, 4], [-8, 2], 'g', linewidth=2.5, label='hyperplane')

plt.xlabel(r'$\phi_1(x)$')
plt.ylabel(r'$\phi_2(x)$')
plt.title(r'$\mathcal{Z}\ space$')
plt.legend(loc='upper left')

plt.savefig('q1.png')
