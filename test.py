# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:29:59 2019

@author: 35764
"""
from alns import ALNS, State
from alns.criteria import HillClimbing
import itertools
import numpy.random as rnd
import networkx as nx
import tsplib95
import tsplib95.distances as distances
import matplotlib.pyplot as plt
import sys


#alns libiary
import warnings
from collections import OrderedDict

import numpy as np
import numpy.random as rnd

from alns.Result import Result
from alns.State import State  # pylint: disable=unused-import
from alns.Statistics import Statistics
from alns.WeigthIndex import WeightIndex
from alns.criteria import AcceptanceCriterion  # pylint: disable=unused-import
from alns.exceptions_warnings import OverwriteWarning
from alns.select_operator import select_operator




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



#Solution state
class TspState(State):
    """
    Solution class for the TSP problem. It has two data members, nodes, and edges.
    nodes is a list of node tuples: (id, coords). The edges data member, then, is
    a mapping from each node to their only outgoing node.
    """

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def copy(self):
        """
        Helper method to ensure each solution state is immutable.
        """
        return TspState(self.nodes.copy(),
                        self.edges.copy())

    def objective(self):
        """
        The objective function is simply the sum of all individual edge lengths,
        using the rounded Euclidean norm.
        """
        return sum(distances.euclidean(node[1], self.edges[node][1])
                    for node in self.nodes)
    
    def to_graph(self):
        """
        NetworkX helper method.
        """
        graph = nx.Graph()

        for node, coord in self.nodes:
            graph.add_node(node, pos=coord)

        for node_from, node_to in self.edges.items():
            graph.add_edge(node_from[0], node_to[0])

        return graph
    
    
    
    
#Destroy operators    
degree_of_destruction = 0.25

def edges_to_remove(state):
    return int(len(state.edges) * degree_of_destruction)    
    
def worst_removal(current, random_state):
    """
    Worst removal iteratively removes the 'worst' edges, that is,
    those edges that have the largest distance.
    """
    destroyed = current.copy()

    worst_edges = sorted(destroyed.nodes,
                         key=lambda node: distances.euclidean(node[1],
                                                              destroyed.edges[node][1]))

    for idx in range(edges_to_remove(current)):
        del destroyed.edges[worst_edges[-idx -1]]

    return destroyed    

def path_removal(current, random_state):
    """
    Removes an entire consecutive subpath, that is, a series of
    contiguous edges.
    """
    destroyed = current.copy()
    
    node_idx = random_state.choice(len(destroyed.nodes))
    node = destroyed.nodes[node_idx]
    
    for _ in range(edges_to_remove(current)):
        node = destroyed.edges.pop(node)

    return destroyed

def random_removal(current, random_state):
    """
    Random removal iteratively removes random edges.
    """
    destroyed = current.copy()
    
    for idx in random_state.choice(len(destroyed.nodes),
                                   edges_to_remove(current),
                                   replace=False):
        del destroyed.edges[destroyed.nodes[idx]]

    return destroyed



#Repair operators    
def would_form_subcycle(from_node, to_node, state):
    """
    from_node：随机排列组合中的下一个点    to_node：前往的点       state:tspsolution类
    Ensures the proposed solution would not result in a cycle smaller
    than the entire set of nodes. Notice the offsets: we do not count
    the current node under consideration, as it cannot yet be part of
    a cycle.
    """
    for step in range(1, len(state.nodes)):
        if to_node not in state.edges:
            return False

        to_node = state.edges[to_node]
        
        if from_node == to_node and step != len(state.nodes) - 1:
            return True

    return False


def greedy_repair(current, random_state):
    """
    current是tspsolution类，有nodes和edges两个属性
    Greedily repairs a tour, stitching up nodes that are not departed
    with those not visited.
    """
    visited = set(current.edges.values())
  
    # This kind of randomness ensures we do not cycle between the same
    # destroy and repair steps every time.
    shuffled_idcs = random_state.permutation(len(current.nodes))   # 0到130的ndarray随机排列组合
    nodes = [current.nodes[idx] for idx in shuffled_idcs]

    while len(current.edges) != len(current.nodes): #.edges边 .nodes点
        node = next(node for node in nodes 
                    if node not in current.edges) 

        # Computes all nodes that have not currently been visited,
        # that is, those that this node might visit. This should
        # not result in a subcycle, as that would violate the TSP
        # constraints.
        unvisited = {other for other in current.nodes
                     if other != node
                     if other not in visited
                     if not would_form_subcycle(node, other, current)}
        d_near = sys.maxsize
        for item in unvisited:
            e_num = item[0]
            d = distance[node[0]][e_num]
            if d < d_near:
                d_near = d
                d_num = item
        
        nearest = d_num
        # Closest visitable node.
        # nearest = min(unvisited, key=lambda other: distances.euclidean(node[1], other[1]))

        current.edges[node] = nearest
        visited.add(nearest)

    return current




#Initial solution
random_state = rnd.RandomState(SEED)
state = TspState(list(pos.items()), {})

initial_solution = greedy_repair(state, random_state)

initial_dis = 0

for a,b in initial_solution.edges.items():
    a=a[0]
    b=b[0]
    initial_dis += distance[a][b]


print("Initial solution objective is {0}.".format(initial_dis))
#print("Initial solution objective is {0}.".format(initial_solution.objective()))




#画图 
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









class ALNS:

    def __init__(self, rnd_state=rnd.RandomState()):
        self._destroy_operators = OrderedDict()
        self._repair_operators = OrderedDict()
        self._rnd_state = rnd_state

    def iterate(self, initial_solution, weights, operator_decay, criterion,
                iterations=10000, collect_stats=True):

        weights = np.asarray(weights, dtype=np.float16)

        self._validate_parameters(weights, operator_decay, iterations)

        current = best = initial_solution

        d_weights = np.ones(len(self.destroy_operators), dtype=np.float16)
        r_weights = np.ones(len(self.repair_operators), dtype=np.float16)

        statistics = Statistics()

        if collect_stats:
            statistics.collect_objective(initial_solution.objective())

        for iteration in range(iterations):
            d_idx = select_operator(self.destroy_operators, d_weights,
                                    self._rnd_state)

            r_idx = select_operator(self.repair_operators, r_weights,
                                    self._rnd_state)

            d_name, d_operator = self.destroy_operators[d_idx]
            destroyed = d_operator(current, self._rnd_state)

            r_name, r_operator = self.repair_operators[r_idx]
            candidate = r_operator(destroyed, self._rnd_state)

            current, weight_idx = self._consider_candidate(best, current,
                                                            candidate, criterion)

            if current.objective() < best.objective():
                best = current

            # The weights are updated as convex combinations of the current
            # weight and the update parameter. See eq. (2), p. 12.
            d_weights[d_idx] *= operator_decay
            d_weights[d_idx] += (1 - operator_decay) * weights[weight_idx]

            r_weights[r_idx] *= operator_decay
            r_weights[r_idx] += (1 - operator_decay) * weights[weight_idx]

            if collect_stats:
                statistics.collect_objective(current.objective())

                statistics.collect_destroy_operator(d_name, weight_idx)
                statistics.collect_repair_operator(r_name, weight_idx)

        return Result(best, statistics if collect_stats else None)

    def add_destroy_operator(self, operator, name=None):
        self._add_operator(self._destroy_operators, operator, name)

    @staticmethod
    def _add_operator(operators, operator, name=None):
        if name is None:
            name = operator.__name__

        if name in operators:
            warnings.warn("The ALNS instance already knows an operator by the"
                          " name `{0}'. This operator will now be replaced with"
                          " the newly passed-in operator. If this is not what"
                          " you intended, consider explicitly naming your"
                          " operators via the `name' argument.".format(name),
                          OverwriteWarning)

        operators[name] = operator


    def add_repair_operator(self, operator, name=None):
        self._add_operator(self._repair_operators, operator, name)


    def _validate_parameters(self, weights, operator_decay, iterations):
        if len(self.destroy_operators) == 0 or len(self.repair_operators) == 0:
            raise ValueError("Missing at least one destroy or repair operator.")

        if not (0 < operator_decay < 1):
            raise ValueError("Operator decay parameter outside unit interval"
                             " is not understood.")

        if any(weight <= 0 for weight in weights):
            raise ValueError("Non-positive weights are not understood.")

        if len(weights) < 4:
            # More than four is not explicitly problematic, as we only use the
            # first four anyways.
            raise ValueError("Unsupported number of weights: expected 4,"
                             " found {0}.".format(len(weights)))

        if iterations < 0:
            raise ValueError("Negative number of iterations.")

    @property
    def destroy_operators(self):
        return list(self._destroy_operators.items())

    @property
    def repair_operators(self):
        return list(self._repair_operators.items())

    def _consider_candidate(self, best, current, candidate, criterion):
        if candidate.objective() < best.objective():
            return candidate, WeightIndex.IS_BEST

        if candidate.objective() < current.objective():
            return candidate, WeightIndex.IS_BETTER

        if criterion.accept(self._rnd_state, best, current, candidate):
            return candidate, WeightIndex.IS_ACCEPTED

        return current, WeightIndex.IS_REJECTED




#Heuristic solution
alns = ALNS(random_state)

alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(path_removal)
alns.add_destroy_operator(worst_removal)

alns.add_repair_operator(greedy_repair)





from abc import ABC, abstractmethod
from alns.State import State  # pylint: disable=unused-import
from numpy.random import RandomState  # pylint: disable=unused-import

class AcceptanceCriterion(ABC):
    @abstractmethod
    def accept(self, rnd, best, current, candidate):
        return NotImplemented


class HillClimbing(AcceptanceCriterion):
    def accept(self, rnd, best, current, candidate):
        return candidate.objective() <= current.objective()






# This is perhaps the simplest selection criterion, where we only accept
# progressively better solutions.
criterion = HillClimbing()
'''
[3,2,1,0.5]最好、局部最优、接受、拒绝的权重
0.8运算衰减参数
criterion接受标准
'''
result = alns.iterate(initial_solution, [3, 2, 1, 0.5], 0.8, criterion,
                      iterations=5000, collect_stats=True)

solution = result.best_state

objective = solution.objective()



print('Best heuristic objective is {0}.'.format(objective))
print('This is {0:.1f}% worse than the optimal solution, which is {1}.'
      .format(100 * (objective - optimal) / optimal, optimal))

_, ax = plt.subplots(figsize=(12, 6))
result.plot_objectives(ax=ax, lw=2)






figure = plt.figure("operator_counts", figsize=(14, 6))
figure.subplots_adjust(bottom=0.15, hspace=.5)
result.plot_operator_counts(figure=figure, title="Operator diagnostics")


draw_graph(solution.to_graph())





#Post-processing
k = 6

def fix_bounds(permutation, start_node, end_node):
    """
    Fixes the given permutation to the start and end nodes, such that
    it connects up to the remainder of the solution.
    """
    return (start_node,) + permutation + (end_node,)


def optimal_subpath(nodes, start_node, end_node):
    """
    Computes the minimum cost subpath from the given nodes, where the 
    subpath is fixed at start_node and end_node.
    """
    def cost(subpath):
        path = fix_bounds(subpath, start_node, end_node)
        
        return sum(distances.euclidean(path[idx][1], path[idx + 1][1])
                   for idx in range(len(path) - 1))

    subpath = min(itertools.permutations(nodes, k), key=cost)

    return fix_bounds(subpath, start_node, end_node)





def post_process(state):
    """
    For each node in the passed-in state, this post processing step 
    computes the optimal subpath consisting of the next k nodes. This
    results in a run-time complexity of about O(n * k!), where n is
    the number of nodes.
    """
    state = state.copy()

    for start_node in state.nodes:
        nodes = []
        node = start_node

        # Determine the next k nodes that make up the subpath starting
        # at this start_node.
        for _ in range(k):
            node = state.edges[node]
            nodes.append(node)
        
        end_node = state.edges[node]

        optimal = optimal_subpath(nodes, start_node, end_node)
        
        # Replace the existing path with the optimal subpath.
        for first, second in zip(optimal, optimal[1:]):
            state.edges[first] = second

    return state



new_solution = post_process(solution)

new_objective = new_solution.objective()

print("New heuristic objective is {0}.".format(new_objective))
print('This is {0:.1f}% worse than the optimal solution, which is {1}.'
      .format(100 * (new_objective - optimal) / optimal, optimal))





draw_graph(new_solution.to_graph())