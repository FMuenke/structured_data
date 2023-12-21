import numpy as np
from structured_data.nodes import Node, GroupOfNodes


class NearestNeighboursGraph:
    def __init__(self, x, k, samples=None):
        if samples is None:
            samples = [None] * len(x.shape[0])
        assert x.shape[0] == len(samples), "Size does not match!"
        self.n_samples = len(samples)
        self.k = k

        self.list_of_nodes = GroupOfNodes([Node(i, x[i, :], samples[i]) for i in range(self.n_samples)])

    def query(self, x_sample, n):
        x = np.array([node.representation for node in self.list_of_nodes])
        distances = np.sqrt(np.sum(np.square(x - x_sample), axis=1))
        sorted_indices = np.argsort(distances)
        return [self.list_of_nodes[i] for i in sorted_indices[:n]]
    
    def _walk_graph(self, nodes, nodes_to_match, n_steps):
        if n_steps <= 0:
            return nodes
        neighbours = GroupOfNodes([])
        for node in nodes:
            node_neighbours, nodes_to_match = node.get_neighbours(nodes_to_match, self.k)
            neighbours += node_neighbours
        return nodes + self._walk_graph(neighbours, nodes_to_match, n_steps - 1)

    def get_neighbours(self, start_index, n_steps):
        nodes = GroupOfNodes([self.list_of_nodes[start_index]])
        nodes_to_match = GroupOfNodes(self.list_of_nodes[:start_index] + self.list_of_nodes[start_index+1:])
        neighbours = self._walk_graph(nodes, nodes_to_match, n_steps)
        return neighbours

    def select_cluster(self, start_index):
        nodes = GroupOfNodes([self.list_of_nodes[start_index]])
        nodes_to_match = GroupOfNodes(self.list_of_nodes[:start_index] + self.list_of_nodes[start_index+1:])
        len_nodes = 0
        while len(nodes) > len_nodes:
            len_nodes = len(nodes)
            nodes.add_neighbours(nodes_to_match, self.k)
            nodes.prune()
        return nodes
