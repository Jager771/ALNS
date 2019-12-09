# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:52:28 2019

@author: 35764
"""


import itertools
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
import sys
from operator import itemgetter
import numpy as np 
import random

w_index = 0
best_distance = sys.maxsize
local_best_distance = sys.maxsize

SEED = 9876
x_train=[]
y_train=[]
pos= {}
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


#画图 
G=nx.Graph()
nodes = point
names = {n for n in range(len(point))}
G.add_nodes_from(names)
fig, ax = plt.subplots(figsize=(12, 6))
func = nx.draw_networkx_nodes(G,pos, node_size=25, with_labels=False)


#模拟退火接受参数
T = 9000  # 起始温度
alpha = 0.995  # T_{k+1} = alpha * T_k方式更新温度
limitedT = 1.  # 最小值的T
iterTime = 2000  # 每个温度下迭代的次数
K = 0.8  # 系数K
p = 0

#评估系数
lenda = 0.8 # 
w= [3, 2, 1, 0.5] 
num_of_method = 0
weight = [1,1,1]    #权重加方法要变
destory_p = [] #各种destory方法的概率
#initial_solution当前解，best_sol全局最优解
degree_of_destruction = 0.25

class TspState():
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def copy(self):
        return TspState(self.nodes.copy(),
                        self.edges.copy())

    def accept_1(self,local_best_distance,result,initial_solution,destoryed):
        # global T,alpha,limitedT,iterTime,K,p
        # w2_index,w3_index,w4_index = 0,0,0    
        global T
        global w_index
        T *= alpha
        if result < local_best_distance:
            w_index = 1
            local_best_distance = result
            initial_solution = destoryed
            return w_index,local_best_distance,initial_solution
        else:
            p = np.exp((local_best_distance - result) / (K * T))
            if random.random() < p:
                w_index = 2
                local_best_distance = result
                initial_solution = destoryed
                return w_index,local_best_distance,initial_solution
            else:
                w_index = 3
                return w_index,local_best_distance,initial_solution
        

    def accept_2(self,best_distance,result,best_sol,destoryed):
        global w_index
        if result < best_distance:
            w_index = 0
            best_distance = result
            best_sol = destoryed
        return w_index,best_distance,best_sol
        
    def edges_to_remove(state):
        return int(len(state.edges) * degree_of_destruction)
    
    
        '''
        加方法要变
        '''
    def choose(self):
        p_buttom = 0 #选择方法时候的底数

        for i in range(len(weight)):
            p_buttom += weight[i]

        p1 = weight[0] / p_buttom
        p2 = weight[1] / p_buttom
        p3 = 1 - p1 -p2
        # np.random.seed(0)
        p = np.array([p1,p2,p3])
     
        '''
        加方法要变
        '''
        
        
        index_of_destory = np.random.choice([0, 1, 2], p = p.ravel())   #加方法要变
        return index_of_destory
    
    
    def update(self,num_of_method,index):
        global weight
        weight[num_of_method] = lenda * weight[num_of_method] + (1 - lenda)* w[index]
        


    def calculate_distance(self,state):
        distance_result = 0
        for a,b in state.edges.items():
            a=a[0]
            b=b[0]
            distance_result += distance[a][b]
        return distance_result
    
    
class r_greedy():
    def greedy_repair(self,current, random_state):
        visited = set(current.edges.values())
      
        #随机保证每次的Greedy不是同一种repair
        shuffled_idcs = random_state.permutation(len(current.nodes))   # 0到130的ndarray随机排列组合
        nodes = [current.nodes[idx] for idx in shuffled_idcs]
    
        while len(current.edges) != len(current.nodes): #当点与边不同的时候
            node = next(node for node in nodes          #选择当前不在edge的点
                        if node not in current.edges) 
    
            # Computes all nodes that have not currently been visited,
            # that is, those that this node might visit. This should
            # not result in a subcycle, as that would violate the TSP
            # constraints.
            unvisited = {other for other in current.nodes
                         if other != node
                         if other not in visited
                         if not self.would_form_subcycle(node, other, current)}
            #unvisited 可以的候选解 1、不等于起始点 2、入度为0 3、
            d_near = sys.maxsize
            for item in unvisited:
                e_num = item[0]
                d = distance[node[0]][e_num]
                if d < d_near:
                    d_near = d
                    d_num = item
            
            nearest = d_num
            # Closest visitable node.
           
    
            current.edges[node] = nearest #选择node的最近点
            visited.add(nearest)
    
        return current


    def would_form_subcycle(self,from_node, to_node, state):
        """
        from_node：随机排列组合中的下一个点    to_node：前往的点       state:tspsolution类
        Ensures the proposed solution would not result in a cycle smaller
        than the entire set of nodes. Notice the offsets: we do not count
        the current node under consideration, as it cannot yet be part of
        a cycle.
        """
        #返回false 接受
        for step in range(1, len(state.nodes)):  #len - 1个，把当前计算的点排除在外，因为不可能形成子循环
            if to_node not in state.edges:      #to_node 出度为0
                return False
    
            to_node = state.edges[to_node]      #若to_node出度为1，则to_node指向其指向的节点
            
            if from_node == to_node and step != len(state.nodes) - 1:   #若
                return True
        
        #for结束后则return False
        return False
#repair的greedy方法
r_g = r_greedy()


#随机种子
random_state = rnd.RandomState(SEED)

state = TspState(list(pos.items()), {})





initial_solution = r_g.greedy_repair(state, random_state)

initial_dis = 0

initial_dis = initial_solution.calculate_distance(initial_solution)


print("Initial solution objective is {0}.".format(initial_dis))
best_distance = initial_dis
local_best_distance = initial_dis
best_sol = initial_solution


#破坏的数量
num_of_destoryed = initial_solution.edges_to_remove()


#初始解画图 
G=nx.Graph()
nodes = point
names = {n for n in range(len(point))}
G.add_nodes_from(names)
for a,b in initial_solution.edges.items():
    a=a[0]
    b=b[0]
    G.add_edge(a,b)
fig, ax = plt.subplots(figsize=(12, 6))
func = nx.draw_networkx(G,pos, node_size=25, with_labels=False)


class d_longest():
    def search_for_longest(self,destoryed):

        s_f_l = {}
        for a,b in destoryed.edges.items():
            a=a[0]
            b=b[0]
            s_f_l[(a,b)]= distance[a][b]
        s_f_l = sorted(s_f_l.items(),key = itemgetter(1), reverse = True)
        return s_f_l
    
    def destroy(self,destoryed):

        _ = self.search_for_longest(destoryed)
        for i in range(num_of_destoryed):
            destoryed.edges.pop(destoryed.nodes[_[i][0][0]])
    
    def solve(self,state):
        destoryed = state.copy()
        self.destroy(destoryed)
        destoryed = r_g.greedy_repair(destoryed, random_state)
        return destoryed
#destory的找最长方法
d_l = d_longest()        

class d_random30():
    def solve(self,state):

        s = [ i for i in range(n)]
        r = random.sample(s,num_of_destoryed)
        destoryed = state.copy()
        for i in range(num_of_destoryed):
            destoryed.edges.pop(destoryed.nodes[r[i]])    
        destoryed = r_g.greedy_repair(destoryed, random_state)
        return destoryed
#destory的d_random30方法
d_r = d_random30()



class d_consective():
    def solve(self,state):
        r = random.randint(0,n-1)
        destoryed = state.copy()
        
        node_to_destory = state.nodes[r] 
        next_node = destoryed.edges[node_to_destory]
        
        for i in range(num_of_destoryed):
            
            destoryed.edges.pop(node_to_destory)    
            
            node_to_destory = next_node
            next_node = destoryed.edges[node_to_destory]
        destoryed = r_g.greedy_repair(destoryed, random_state)
        return destoryed
d_c = d_consective()




if __name__ == '__main__':
    for i in range(5000):
        index_of_destory = initial_solution.choose()
        '''
        加方法要变
        '''
        if index_of_destory == 0:
            destoryed = d_l.solve(initial_solution)
    
        elif index_of_destory == 1:
            destoryed = d_r.solve(initial_solution)
            
        elif index_of_destory == 2:
            destoryed = d_c.solve(initial_solution)
            
            
        result = destoryed.calculate_distance(destoryed)
        index,local_best_distance,initial_solution=initial_solution.accept_1(local_best_distance,result,initial_solution,destoryed)
        index,best_distance,best_sol=initial_solution.accept_2(best_distance,result,best_sol,destoryed)
        
        destoryed.update(index_of_destory,index)

    

    #画最终解的图 
    G=nx.Graph()
    nodes = point
    names = {n for n in range(len(point))}
    G.add_nodes_from(names)
    
    
    for a,b in best_sol.edges.items():
        a=a[0]
        b=b[0]
        G.add_edge(a,b)
    
    # G.add_edge(best_route[-1],best_route[-2])    
    fig, ax = plt.subplots(figsize=(12, 6))
    func = nx.draw_networkx(G,pos, node_size=25, with_labels=False)
    
    
    print('Best heuristic objective is {0}.'.format(best_distance))
    print('This is {0:.1f}% worse than the optimal solution, which is {1}.'
          .format(100 * (best_distance - optimal) / optimal, optimal))





