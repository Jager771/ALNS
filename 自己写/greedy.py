# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:07:57 2019

@author: 35764
"""


def greedy(num):     
    global distance
    sum = 0.0
    dis = [[0 for col in range(num)] for raw in range(num)]
    for i in range(num):
        for j in range(num):
            dis[i][j] = distance[i][j]
                
    visited = []
    #进行贪婪选择——每次都选择距离最近的
    id = 0
    for i in range(num):
        for j in range(num):
            dis[j][id] = sys.maxsize  #将已经走过的设置为最大值
        minvalue = min(dis[id])
        if i != num:
            sum += minvalue
        visited.append(id)
        id = dis[id].index(minvalue)
    sum += distance[0][visited[n-1]]
    return visited



