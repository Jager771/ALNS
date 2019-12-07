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
    optimal = _[3]

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
















#Heuristic solution
alns = ALNS(random_state)

alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(path_removal)
alns.add_destroy_operator(worst_removal)

alns.add_repair_operator(greedy_repair)



# This is perhaps the simplest selection criterion, where we only accept
# progressively better solutions.
criterion = HillClimbing()

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