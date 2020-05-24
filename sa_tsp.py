#!/usr/bin/python 
# -*-coding:utf-8 -*-
__author__ = '99K'

from sko.SA import SA_TSP
import time
import numpy as np
import matplotlib.pyplot as plt


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


problem_set = 'kroC100'
tmax = 100
tmin = 1
setting = '%s_SA' % problem_set
start = time.process_time()

nodeMap, nodeDis = readData('tsp_data/%s.tsp' % problem_set)
opt = readOpt('tsp_data/%s.opt.tour' % problem_set)
opt_distance = calculateTotalDis(opt)
opt = np.concatenate([opt, [opt[0]]])

fig, ax = plt.subplots(1, 3)
sa_tsp = SA_TSP(func=calculateTotalDis, x0=range(len(nodeDis)), T_max=tmax, T_min=tmin, L=10 * len(nodeDis))
best_points, best_distance = sa_tsp.run()
best_points_ = np.concatenate([best_points, [best_points[0]]])
ax[0].plot([nodeMap[p][0] for p in best_points_], [nodeMap[p][1] for p in best_points_], 'o-r')
ax[0].set_title('SA solution')
finish = time.process_time()
print('time cost', finish-start)
ax[1].plot(sa_tsp.best_y_history)
ax[1].set_title('Distance Curve')
ax[2].plot([nodeMap[p][0] for p in opt], [nodeMap[p][1] for p in opt], 'o-r')
ax[2].set_title('Optimal solution')
fig.suptitle('Exp. %s' % setting)
plt.savefig('figures/%s_tmax%dtmin%dtime%.1f.jpg' % (setting, tmax, tmin, finish-start))
plt.show()

