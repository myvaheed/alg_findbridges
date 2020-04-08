import networkx as nx
import numpy as np
from collections import deque
from math import log2, floor, ceil
from enum import Enum

class Stack :
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()
    
    def last(self): 
        return self.items[-1]

    def is_empty(self):
        return (len(self.items) == 0)


TEST_MODE = False
NUM_MAX_RAND = 2**64
if TEST_MODE:
    NUM_MAX_RAND = 100

class SortMeth(Enum):
        RADIX = 0,
        TRIMSOTR = 1,
        BUCKET = 2

class BridgeRandomizedSt:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.sorted_edges = []
        self.used = np.zeros(graph.number_of_nodes())
        self.stack = Stack()
        np.random.seed(1234)
        for (_, _, w) in self.graph.edges(data=True):
            w['weight'] = -1

    @staticmethod
    def next_randint():
        return np.random.randint(0, NUM_MAX_RAND, size=1, dtype='uint64')[0]
 
    def dfs(self, v):
        root = -1
        new_invoke = True
        self.stack.push((v, root, 0, 0))    
        while not self.stack.is_empty():
            if new_invoke == True:
                v, p, _, _ = self.stack.last()
                self.used[v] = 1
                new_invoke = False
                for i in range(len(self.graph[v])):
                    u = list(self.graph[v])[i]
                    if u == p:
                        continue
                    if not self.used[u]:
                        self.stack.push((u, v, p, i))
                        new_invoke = True
                        break
                    elif self.graph[v][u]['weight'] == -1:
                        self.graph[v][u]['weight'] = self.next_randint()
                
            else:
                _, v, p, ni = self.stack.pop()

                # if (v == root):
                #      continue
                for i in range(ni + 1, len(self.graph[v])):
                    u = list(self.graph[v])[i]
                    if u == p:
                        continue
                    if not self.used[u]:
                        self.stack.push((u, v, p, i))
                        new_invoke = True
                        break
                    elif self.graph[v][u]['weight'] == -1:
                        self.graph[v][u]['weight'] = self.next_randint()
            
            if new_invoke:
                continue
            if p == root: #p-root
                return
            res = 0
            for u in self.graph[v]:
                if u == p:
                    continue
                w = int(self.graph[v][u]['weight'])
                res ^= w
            self.graph[v][p]['weight'] = res

    @staticmethod
    def bucked_sort(arraylist: list) -> list:
        import operator
        from functools import reduce

        n = len(arraylist)
        B = [list() for _ in range(n)]

        for item in arraylist:
            index = floor(n * item[2]['weight']/float(NUM_MAX_RAND))
            B[index].append(item)
        
        res = []
        for deq in B:
            res.append(sorted(deq, key=lambda t: t[2]['weight']))
        
        total = []
        for r in res:
            for el in r:
                total.append(el)
#         return list(reduce(operator.add, res))
        return total


    @staticmethod
    def radix_sort(array: list) -> list:
        array = list(filter(lambda x: len(x) != 0, array))
        B = [0] * len(array)
        
        SS = 2**16 - 1

        for i in range(0, ceil(log2(NUM_MAX_RAND)/4)):
            # counting sort          
            counters = [0] * SS

            for item in array:
                w = int(item[2]['weight'])
                num = (w >> i*16) & SS
                counters[num] += 1   
            
            for j in range(1, SS):
                counters[j] += counters[j-1]

            for j in range(SS):
                counters[j] -= 1

            for item in array[::-1]:
                w = int(item[2]['weight'])
                num = (w >> i*16) & SS

                B[counters[num]] = item
                counters[num] -= 1
                
            array = B
        return array

    def run(self, sort: SortMeth):
        for i in range(self.graph.number_of_nodes()):
            if not self.used[i]:
                self.dfs(i)

        if sort == SortMeth.RADIX:
            self.sorted_edges = self.radix_sort(self.graph.edges(data=True))
        elif sort == SortMeth.TRIMSOTR:
            self.sorted_edges = sorted(self.graph.edges(data=True), key=lambda t: t[2]['weight'])
        else:
            self.sorted_edges = self.bucked_sort(self.graph.edges(data=True))

def filter_edges_to_bridges(sorted_edges) -> list:
    graph = nx.from_edgelist(sorted_edges)
    bridges = []
    n_comp = nx.number_connected_components(graph)
    while True:
        potential_bridge =  sorted_edges.pop(0)
        graph.remove_edge(*potential_bridge[:2])
        n_next = int(nx.number_connected_components(graph))
        graph.add_edge(*potential_bridge[:2])
        if n_next == n_comp or len(sorted_edges) == 0:
            break
        bridges.append(potential_bridge[:2])
    del graph
    return bridges

edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5)]
G = nx.from_edgelist(edges)

# if not TEST_MODE:
G = nx.generators.erdos_renyi_graph(10, 0.2)

print(len(list(nx.bridges(G))))
brDetSt = BridgeRandomizedSt(G)
brDetSt.run(SortMeth.RADIX)
print(len(filter_edges_to_bridges(brDetSt.sorted_edges)))

import unittest
class TestNotebook(unittest.TestCase):
    
    def test_radix_sort(self):
        g = nx.generators.erdos_renyi_graph(1000, 0.01)
        
        # edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5)]
        # g = nx.from_edgelist(edges)

        for (_, _, w) in g.edges(data=True):
            w['weight'] = BridgeRandomizedSt.next_randint()
        sorted_edges = BridgeRandomizedSt.radix_sort(g.edges(data=True))
        wi = sorted_edges[0][2]['weight']
        for (_, _, w) in sorted_edges:
            if wi > w['weight']:
                self.fail("radix is not correct")
            wi = w['weight']

    def test_bucket_sort(self):
        g = nx.generators.erdos_renyi_graph(1000, 0.01)

        edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (3, 5), (4, 5)]
        g = nx.from_edgelist(edges)

        for (_, _, w) in g.edges(data=True):
            w['weight'] = BridgeRandomizedSt.next_randint()
        sorted_edges = BridgeRandomizedSt.bucked_sort(g.edges(data=True))
        wi = sorted_edges[0][2]['weight']
        for (_, _, w) in sorted_edges:
            if wi > w['weight']:
                self.fail("bucket is not correct")
            wi = w['weight']

if __name__ == "__main__":
    unittest.main()
    pass