# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:50:16 2019

@author: 35764
"""

import itertools
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
import sys
import random
import copy

# runtime = int(input("迭代次数："))
pos= {}
SEED = 9876
x_train=[]
y_train=[]



#读取数据
with open('D:\\GIT\\ALNS\\xqf131.tsp.txt') as f:
    data = f.readlines()
    point = data[8:139]
    for i in range(len(point)):
        _ = point[i].split()
        point[i] = (int(_[1]),int(_[2]))
        # pos[i+1]=point[i]
        pos[i]=point[i]
        x_train.append(int(_[1]))
        y_train.append(int(_[2]))

with open('D:\\GIT\\ALNS\\xqf131.tour.txt') as f:
    best_solution = f.readlines()
    solution = best_solution[1]
    _ = solution.split(' ')
    optimal = int(_[3])

print('Total optimal tour length is {0}.'.format(optimal))



#构建初始参考距离矩阵
n=len(point)
distance = [[0 for col in range(n)] for raw in range(n)]
def getdistance():
    for i in range(n):
        for j in range(n):
            x = pow(x_train[i] - x_train[j], 2)
            y = pow(y_train[i] - y_train[j], 2)
            distance[i][j] = pow(x + y, 0.5)
    for i in range(len(point)):
        for j in range(len(point)):
            if distance[i][j] == 0:
                distance[i][j] = sys.maxsize
getdistance()




#initial
#最佳路径
best_route = []
#最佳距离，开始时为系统最大值
best_distance = sys.maxsize
current_route = []
current_distance = 0.0

#禁忌表
tabu_list = []
tabu_time = []
#当前禁忌对象数量
current_tabu_num = 0
#禁忌长度，即禁忌期限
tabu_limit = 300
#候选集
candidate = [[0 for col in range(n)] for raw in range(300)]
candidate_distance = [0 for col in range(300)]




#画点分布图 
G=nx.Graph()
nodes = point
names = {n for n in range(len(point))}
G.add_nodes_from(names)
fig, ax = plt.subplots(figsize=(12, 6))
func = nx.draw_networkx_nodes(G,pos, node_size=25, with_labels=False)



def greedy(num,start):     
    global distance
    sum = 0.0
    dis = [[0 for col in range(num)] for raw in range(num)]
    for i in range(num):
        for j in range(num):
            dis[i][j] = distance[i][j]
    visited = []
    #进行贪婪选择——每次都选择距离最近的
    id = start
    for i in range(num):
        for j in range(num):
            dis[j][id] = sys.maxsize  #将已经走过的设置为最大值
        minvalue = min(dis[id])
        if i != num-1:
            sum += minvalue
        visited.append(id)
        id = dis[id].index(minvalue)
    visited.append(start)
    sum += distance[0][visited[n-1]]
    return visited,sum
best_route,best_distance = greedy(n,0)
current_route = best_route


#画初始解的图 
G=nx.Graph()
nodes = point
names = {n for n in range(len(point))}
G.add_nodes_from(names)
for _ in range(n-1):
    a=best_route[_]
    b=best_route[_+1]
    G.add_edge(a,b)
G.add_edge(best_route[-1],best_route[-2])    
fig, ax = plt.subplots(figsize=(12, 6))
func = nx.draw_networkx(G,pos, node_size=25, with_labels=False)



class alns:
    def __init__(self):
        self.des_wei = [1,1,1,1]
        self.rep_wei = [1,1,1,1]



    def choose(self):
        pass









class tabu:
    def tabu_get_candidate(self):
        global best_route
        global best_distance
        global current_tabu_num
        global current_distance
        global current_route
        global tabu_list
        global n

        candidate_distance = [0 for col in range(300)]
        #存储两个交换的位置
        exchange_position = []
        temp = 0
        #随机选取邻域
        while True:
            current = random.sample(range(0, n), 2)
            #print(current)
            if current not in exchange_position:
                exchange_position.append(current)
                for i in range(30): #进行30次2-opt
                    candidate[temp] = self.exchange(current[0], current[1], current_route)
                if candidate[temp] not in tabu_list:
                    candidate_distance[temp] = self.cacl_best(candidate[temp])
                    temp += 1
                if temp >= 300:
                    break
                
        
        #得到候选解中的最优解
        candidate_best = min(candidate_distance)
        best_index = candidate_distance.index(candidate_best)
        current_distance = candidate_best
        current_route = copy.copy(candidate[best_index])
        
        #与当前最优解进行比较 
        if current_distance < best_distance:
            best_distance = current_distance
            best_route = copy.copy(current_route)
        
        #加入禁忌表
        tabu_list.append(candidate[best_index])
        tabu_time.append(tabu_limit)
        current_tabu_num += 1        
        

    #更新禁忌表以及禁忌期限
    def update_tabu(self):
        global current_tabu_num
        global tabu_time
        global tabu_list
        
        del_num = 0
        temp = [0 for col in range(n)]
        #更新步长
        tabu_time = [x-1 for x in tabu_time]
        #如果达到期限，释放
        for i in range(current_tabu_num):
            if tabu_time[i] == 0:
                del_num += 1
                tabu_list[i] = temp
               
        current_tabu_num -= del_num        
        while 0 in tabu_time:
            tabu_time.remove(0)
        
        while temp in tabu_list:
            tabu_list.remove(temp)
    
    #交换数组两个元素
    def exchange(self,index1, index2, arr):
        current_list = copy.copy(arr)
        current = current_list[index1]
        current_list[index1] = current_list[index2]
        current_list[index2] = current
        return current_list
            

    #计算总距离
    def cacl_best(self,rou):
        sumdis = 0.0
        for i in range(n-1):
            sumdis += distance[rou[i]][rou[i+1]]
        sumdis += distance[rou[n-1]][rou[0]]     
        return sumdis        
        
    def solve(self):
        self.tabu_get_candidate()
        self.update_tabu()
         
    
class d_longest:
    def find_longest(self):
        pass
        

    
    
    
t = tabu()
d = d_longest()
for i in range(5000):
    t.solve()
        
        
        
        
        
        
        
        
        
        
        
        