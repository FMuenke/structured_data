import numpy as np


class Node:
    def __init__(self, index, representation, sample=None):
        self.id = index
        self.index = index
        self.sample = sample
        self.representation = representation
        self.neighbours = None

    def __str__(self):
        return str(self.index)

    def get_neighbours(self, list_of_nodes, k):
        if len(list_of_nodes) == 0:
            return GroupOfNodes([]), GroupOfNodes([])
        x = np.array([node.representation for node in list_of_nodes])
        distances = np.sqrt(np.sum(np.square(x - self.representation), axis=1))
        sorted_indices = np.argsort(distances)
        neighbours = GroupOfNodes([list_of_nodes[i] for i in sorted_indices[:k]])
        remaining_nodes = GroupOfNodes([list_of_nodes[i] for i in sorted_indices[k:]])
        return neighbours, remaining_nodes
    
    def init_neighbours(self, list_of_nodes, k):
        neighbours, _ = self.get_neighbours(list_of_nodes, k)
        self.neighbours = neighbours

    def get_repr(self):
        return self.representation

    def load_data(self):
        if self.sample is not None:
            return self.sample.load_data()
        return self.representation
    
    def load_y(self):
        if self.sample is not None:
            return self.sample.load_y()
        return self.index
        

class GroupOfNodes:
    def __init__(self, list_of_nodes):
        self.list_of_nodes = list_of_nodes

    def __getitem__(self, i):
        return self.list_of_nodes[i]

    def __iter__(self):
        return self.list_of_nodes.__iter__()

    def __add__(self, group_of_nodes):
        return GroupOfNodes(self.list_of_nodes + group_of_nodes.list_of_nodes)
    
    def add(self, node):
        self.list_of_nodes.append(node)
    
    def add_list(self, group_of_nodes):
        for node in group_of_nodes:
            self.list_of_nodes.append(node)
    
    def __sub__(self, group_of_nodes):
        unique_ids = [node.index for node in group_of_nodes.unique()]
        remaining_nodes = [node for node in self.list_of_nodes if node.index not in unique_ids]
        return GroupOfNodes(remaining_nodes)

    def __len__(self):
        return len(self.list_of_nodes)

    def prune(self):
        self.list_of_nodes = self.unique()

    def index(self):
        return self.list_of_nodes[0].index

    def unique(self):
        list_of_ids, list_of_nodes = [], []
        for node in self.list_of_nodes:
            if node.index in list_of_ids:
                continue
            list_of_ids.append(node.index)
            list_of_nodes.append(node)
        return list_of_nodes

    def get_neighbours(self, nodes_to_match, k):
        neigbouring_nodes = GroupOfNodes([])
        for node in self.list_of_nodes:
            neighbours, _ = node.get_neighbours(nodes_to_match, k)
            neigbouring_nodes += neighbours
        return neigbouring_nodes

    def add_neighbours(self, nodes_to_match, k):
        if len(nodes_to_match) == 0:
            return
        neigbouring_nodes = self.get_neighbours(nodes_to_match, k)
        self.add_list(neigbouring_nodes)

    def get_x(self):
        return np.array([node.representation for node in self.list_of_nodes])
    
    def get_samples(self):
        return [node.sample for node in self.list_of_nodes]
    
    def get_repr(self):
        return self.get_x()
    
    def mean(self):
        return np.mean(self.get_x(), axis=0)
    
    def min(self):
        return np.min(self.get_x(), axis=0)
    
    def max(self):
        return np.max(self.get_x(), axis=0)
    
    def std(self):
        return np.std(self.get_x(), axis=0)
    
    def summarize(self):
        sum_index = self.list_of_nodes[0].index
        sum_representation = self.mean()
        sum_sample = self.list_of_nodes[0].sample
        return Node(sum_index, sum_representation, sample=sum_sample)
    
    def aggregate(self, func=np.mean):
        return func(self.get_x(), axis=0)
