#!/usr/bin/python 
# -*-coding:utf-8 -*-
__author__ = '99K'

import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA
from sko.GA import GA_TSP
import time


def readOpt(filename):
    path = []
    with open(filename) as f:
        for line in f:
            if not line.strip('\n').isnumeric():
                continue
            if '-1' in line or 'EOF' in line:
                break
            node = int(line.strip('\n')) - 1
            path.append(node)
    return path


def readData(filename):
    nodeMap = []
    pts_num = 0
    with open(filename) as f:
        for line in f:
            if not line.strip('\n').split()[0].isnumeric():
                continue
            if 'EOF' in line:
                break
            line = line.strip('\n').split()
            node, x, y = int(line[0]) - 1, int(line[1]), int(line[2])
            
            nodeMap.append([x, y])
            pts_num += 1
    nodeDis = np.zeros((len(nodeMap), len(nodeMap)))
    
    def euleDis(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    for i in range(len(nodeMap)):
        for j in range(len(nodeMap)):
            nodeDis[i][j] = euleDis(nodeMap[i], nodeMap[j])
    return nodeMap, nodeDis


def calculateTotalDis(path):
    totalDis = 0
    for i in range(len(path) - 1):
        totalDis += nodeDis[path[i]][path[i + 1]]
    totalDis += nodeDis[path[-1]][path[0]]
    return totalDis


'''
    func : function
        The func you want to do optimal.
        It inputs a candidate solution(a routine), and return the costs of the routine.
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
'''
problem_set = 'kroC100'
size_pop = 30
prob_mut = 0.5
setting = '%s_A2' % problem_set
start = time.process_time()

nodeMap, nodeDis = readData('tsp_data/%s.tsp' % problem_set)
opt = readOpt('tsp_data/%s.opt.tour' % problem_set)
opt_distance = calculateTotalDis(opt)
opt = np.concatenate([opt, [opt[0]]])

fig, ax = plt.subplots(2, 2)
ga_tsp = GA_TSP(func=calculateTotalDis, n_dim=len(nodeDis), size_pop=size_pop, max_iter=2000, prob_mut=prob_mut)
best_points, best_distance = ga_tsp.run(1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
ax[0][0].plot([nodeMap[p][0] for p in best_points_], [nodeMap[p][1] for p in best_points_], 'o-r')
ax[0][0].set_title('GA Initial solution')
for itr in range(100):
    best_points, best_distance = ga_tsp.run(100)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    # ax[itr].plot([nodeMap[p][0] for p in best_points_], [nodeMap[p][1] for p in best_points_], 'o-r')
    print('itr', itr * 100, '. distance %.2f' % best_distance, 'dif to opt %.2f' % (best_distance - opt_distance))
    if best_distance < opt_distance + 0.01:
        break
finish = time.process_time()
print('time cost', finish-start)
ax[0][1].plot([nodeMap[p][0] for p in best_points_], [nodeMap[p][1] for p in best_points_], 'o-r')
ax[0][1].set_title('GA solution')
ax[1][0].plot(ga_tsp.generation_best_Y)
ax[1][0].set_title('Distance Curve')
ax[1][1].plot([nodeMap[p][0] for p in opt], [nodeMap[p][1] for p in opt], 'o-r')
ax[1][1].set_title('Optimal solution')
fig.suptitle('Exp. %s' % setting)
plt.savefig('figures/%s_pop%dmut%.1fitr%dtime%.1f.jpg' % (setting, size_pop, prob_mut, itr * 100, finish-start))
plt.show()
