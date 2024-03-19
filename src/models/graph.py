import numpy as np 
import random 
from typing import Tuple, List
from queue import Queue

class Graph:

    def __init__(self, num_rows, num_cols, shuffle=True):
        self.num_rows = num_rows
        self.num_cols = num_cols 
        self.num_nodes = num_rows * num_cols
        self.num_directions = 4 # 0 for staying still, 1 for going up, 2 for right, 3 for left and 4 for down  

        self.edges = [[] for i in range(self.num_nodes)]
        self.directions = [[] for i in range(self.num_nodes)]

        self.node_idx = None 
        self.createGrid(shuffle=shuffle)

    def insertEdge(self, node_1, node_2, direction):
        self.edges[node_1].append(node_2)
        self.directions[node_1].append(direction)
        # if direction != 0: # if not self edge
        self.edges[node_2].append(node_1)
        self.directions[node_2].append(self.num_directions - direction - 1)

    def createGrid(self, shuffle=True):
        node_list = [i for i in range(self.num_nodes)]
        if shuffle:
            random.shuffle(node_list)
            # start with 0
            zero_idx = node_list.index(0)
            node_list[0], node_list[zero_idx] = 0, node_list[0]
        # populate grid with random nodes
        self.node_idx = np.array(node_list).reshape((self.num_rows, self.num_cols))
        # create graph
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # insert downward edge if possible
                if i < self.num_rows - 1:
                    node_1, node_2 = self.node_idx[i][j], self.node_idx[i + 1][j]
                    self.insertEdge(node_1, node_2, 3)
                # insert rightward edge if possible
                if j < self.num_cols - 1:
                    node_1, node_2 = self.node_idx[i][j], self.node_idx[i][j + 1]
                    self.insertEdge(node_1, node_2, 1)
                # insert edge to self
                # node = self.node_idx[i][j]
                # self.insertEdge(node, node, 0)

    def printGrid(self): 
        for i in range(self.num_rows):
            for j in range(self.num_cols):
               print(self.node_idx[i][j], end='')
               if j < self.num_cols - 1:
                print(' --------- ', end='')
            print()
            if i < self.num_rows - 1:
                for k in range(4):
                    for l in range(self.num_cols):
                        print('|', end='           ')
                    print()


    def randomWalk(self, num_samples, num_hidden, num_batches):
        """
        This function generates num_samples sequences per batch
        """
        X_train, y_train = [], []
        cur_node = 0
        # select num_hidden paths from current graph
        hidden_paths = []
        # print('Creating hidden paths')
        nodes_in_hidden_paths = set()
        while len(hidden_paths) < num_hidden:
            # pick random node 
            start_node = random.randint(0, self.num_nodes - 1)
            # pick one of it's neighbors
            end_node = random.choice(self.edges[start_node])        
            if start_node not in nodes_in_hidden_paths and \
                end_node not in nodes_in_hidden_paths and \
                start_node != 0 != end_node and start_node != end_node:
                hidden_paths.append([start_node, end_node])
                nodes_in_hidden_paths.add(start_node)
                nodes_in_hidden_paths.add(end_node)
        # print('hidden paths are:')
        # print(hidden_paths)
        # print('Performing random walk over non-hidden paths')
        for i in range(num_batches):
            # Do a random walk while skipping hidden paths
            X, y = np.zeros((num_samples, self.num_directions)), np.zeros((num_samples, self.num_nodes))
            for i in range(num_samples):
                # pick a random node to transition to
                nxt_idx = np.random.randint(0, len(self.edges[cur_node]))
                while [cur_node, self.edges[cur_node][nxt_idx]] in hidden_paths or \
                    [self.edges[cur_node][nxt_idx], cur_node] in hidden_paths:
                    nxt_idx = np.random.randint(0, len(self.edges[cur_node]))

                target = np.zeros((self.num_nodes,))
                target[self.edges[cur_node][nxt_idx]] = 1
        
                inp = np.zeros((self.num_directions,))
                inp[self.directions[cur_node][nxt_idx]] = 1
        
                X[i], y[i] = inp, target 
                cur_node = self.edges[cur_node][nxt_idx]
            X_train.append(X)
            y_train.append(y)
        return np.array(X_train), np.array(y_train), hidden_paths
    
    def one_hot_encode(self, list: List, num_items: int) -> List[int]:
        for i in range(len(list)):
            item = [0 for j in range(num_items)]
            item[list[i]] = 1
            list[i] = item

    def bfs(self, start, target, hidden_paths) -> Tuple[List, List]:
        """
        This function performs BFS from start node to target node and returns the paths as a sequence of observations
        """
        parent, discovered = dict(), set()
        discovered.add(start)
        node_q = Queue()
        node_q.put(start)
        target_found = False
        while not node_q.empty() and target_found is False:
            cur_node = node_q.get()
            for child_node in self.edges[cur_node]:
                if [cur_node, child_node] in hidden_paths or [child_node, cur_node] in hidden_paths:
                    continue 
                if child_node not in discovered:
                    node_q.put(child_node)
                    discovered.add(child_node) 
                    parent[child_node] = cur_node
                    if child_node == target:
                        target_found = True
                        break
                    
        cur_node = target 
        directions, observations = [], []
        cur_node = target 

        while parent.get(cur_node) is not None:
            cur_parent = parent[cur_node]
            idx = self.edges[cur_parent].index(cur_node)
            direction_from_parent = self.directions[cur_parent][idx]
            directions.append(direction_from_parent)
            observations.append(cur_node)
            cur_node = cur_parent
        # add edge from 0 
        idx = self.edges[cur_node]
        directions.reverse(); observations.reverse()
        return directions, observations

    def generateTest(self, hidden_paths, pad=False):
        """
        This function performs BFS from start node (0) to all the nodes in hidden_paths
        """
        X_test, y_test = [], []
        for i in range(len(hidden_paths)):       
            start_node, end_node = hidden_paths[i]
            end_node_idx = self.edges[start_node].index(end_node)
            cur_direction = self.directions[start_node][end_node_idx]
            directions, observations = self.bfs(0, start_node, hidden_paths)
            directions.append(cur_direction)
            observations.append(end_node)
            self.one_hot_encode(directions, self.num_directions)
            self.one_hot_encode(observations, self.num_nodes)
            X_test.append(directions)
            y_test.append(observations)
        if pad:
            # pad test_sequences to generate array of similar shape
            max_len = max([len(x) for x in X_test])
            for i in range(len(X_test)):
                while max_len - len(X_test[i]) > 0:
                    X_test[i].insert(0, [0] * self.num_directions)
                while max_len - len(y_test[i]) > 0:
                    y_test[i].insert(0, [0] * self.num_nodes)
        
        return np.array(X_test), np.array(y_test)
