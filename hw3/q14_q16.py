'''
# Machine Learning Techniques - Hw3 Q.14 to Q.16

## Experiments with Decision Tree

Author: Cheng-Shih Wong
Email:  r04945028@ntu.edu.tw
'''

from __future__ import print_function

import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import pygraphviz
import networkx as nx

import hw3_model as md

def findLeaves(root, leaves):
  if root.isLeaf: return
  
  if root.childs[0].isLeaf:
    leaves.append((root, 0))
  if root.childs[1].isLeaf:
    leaves.append((root, 1))

  findLeaves(root.childs[0], leaves)
  findLeaves(root.childs[1], leaves)

def draw_tree(root, G, nid, pnid):
  if root.isLeaf:
    G.add_node(nid, label='{:.3f}'.format(root.val))
  else:
    G.add_node(nid, label='x_{} < {:.3f}'.format(root.split_feat, root.split_val))
    draw_tree(root.childs[0], G, nid*2, nid)
    draw_tree(root.childs[1], G, nid*2+1, nid)

  if pnid:
    G.add_edge(pnid, nid, label='{}'.format('True' if pnid*2==nid else 'False'))

if __name__ == '__main__':
  if len(sys.argv)!=3:
    print('Usage: python q14_q16.py <train_data> <test_data>')
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

  # Train Decision Tree
  model = md.dtree()
  model.fit(X, y)

  # Draw tree
  G = nx.DiGraph()
  draw_tree(model, G, 1, 0)
  nx.drawing.nx_agraph.write_dot(G, 'Q14.dot')

  # Run experiments
  print('Question 15')

  print('Ein = {:.5f}'.format(1-md.accuracy(y, model.predict(X))))
  print('Eout = {:.5f}'.format(1-md.accuracy(testy, model.predict(testX))))
  print()

  print('Qeustion 16')
  leaves = []

  findLeaves(model, leaves)
  eins = []
  eouts = []

  for (parent, lid) in leaves:
    oleaf = parent.childs[lid]
    parent.childs[lid] = parent.childs[lid^1]

    eins.append(1-md.accuracy(y, model.predict(X)))
    eouts.append(1-md.accuracy(testy, model.predict(testX)))

    parent.childs[lid] = oleaf
  
  print('{:>10}{:>10}'.format('Eins', 'Eouts'))
  for ein, eout in zip(eins, eouts):
    print('{:10.5f}{:10.5f}'.format(ein, eout))

  print()
