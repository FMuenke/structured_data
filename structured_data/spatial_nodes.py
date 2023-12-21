import numpy as np
from structured_data.nodes import Node, GroupOfNodes


class SpatialNode(Node):
    def __init__(self, index, coordinates, representation, sample=None):
        repr_node = Node(index, representation, sample)
        Node.__init__(self, index, coordinates, repr_node)

    def repr_node(self):
        return self.sample
    
    def get_repr(self):
        return self.sample.representation
    
    def to_node(self):
        return self.sample
    

class GroupOfSpatialNodes(GroupOfNodes):
    def __init__(self, list_of_spatial_nodes):
        GroupOfNodes.__init__(self, list_of_spatial_nodes)
    
    def summarize(self):
        sum_index = self.list_of_nodes[0].index
        coordinate = np.mean(np.array([node.representation for node in self.list_of_nodes]), axis=0)
        representation = np.mean(np.array([node.get_repr() for node in self.list_of_nodes]), axis=0)
        sum_sample = self.list_of_nodes[0].sample
        return SpatialNode(sum_index, coordinate, representation, sum_sample)
    
    def to_grp_of_nodes(self):
        return GroupOfNodes([node.to_node() for node in self.list_of_nodes])