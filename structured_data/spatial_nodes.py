import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MiniBatchKMeans

from structured_data.grid import Grid
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
    
    def overwrite_representation(self, repr):
        return SpatialNode(self.index, self.x, repr, self.sample)


class GroupOfSpatialNodes(GroupOfNodes):
    def __init__(self, list_of_spatial_nodes):
        super().__init__(list_of_spatial_nodes)
    
    def get_coords(self):
        return self.get_x()
    
    def mean_coords(self):
        return np.mean(self.get_coords(), axis=0)
    
    def summarize(self, summary_func=np.mean):
        sum_index = self.list_of_nodes[0].index
        coordinate = np.mean(np.array([node.representation for node in self.list_of_nodes]), axis=0)
        representation = summary_func(np.array([node.get_repr() for node in self.list_of_nodes]), axis=0)
        sum_sample = self.list_of_nodes[0].sample
        return SpatialNode(sum_index, coordinate, representation, sum_sample)
    
    def to_grp_of_nodes(self):
        return GroupOfNodes([node.to_node() for node in self.list_of_nodes])
    
    def get_samples(self):
        grp = self.to_grp_of_nodes()
        return grp.get_samples()
    
    def get_repr(self):
        return np.array([node.get_repr() for node in self.list_of_nodes])
    
    def query_by_coords(self, coords, n, dst_min=None):
        distances = np.sqrt(np.sum(np.square(self.get_coords() - coords), axis=1))
        if dst_min is not None:
            distances[distances > dst_min] = np.inf
        sorted_indices = np.argsort(distances)
        geo_nodes = [self.list_of_nodes[i] for i in sorted_indices[:n] if distances[i] < np.inf]
        return GroupOfSpatialNodes(geo_nodes)

    def cut(self, coords, radius):
        return self.query_by_coords(coords, len(self), radius)
    
    def downsample_by_grid(self, grid_size):
        coords = self.get_coords()
        grid = Grid(coords, grid_size)
        list_of_indices = grid.group_points(coords)
        new_nodes = GroupOfSpatialNodes([])
        for list_i in list_of_indices:
            if len(list_i) == 0:
                continue
            grp = GroupOfSpatialNodes([self.list_of_nodes[i] for i in list_i])
            new_nodes.add(grp.summarize())
        return new_nodes
    
    def get_coverage(self, grid_size=4):
        coords = self.get_coords()
        grid = Grid(coords, grid_size)
        return len(grid.group_points(coords))
    
    def make_sub_grps(self, y):
        assert len(y) == len(self), "Assignement Y does not match number of nodes"
        y_mapping = {y_unique: i for i, y_unique in enumerate(np.unique(y))}
        list_of_grps = [GroupOfSpatialNodes([]) for _ in np.unique(y)]
        for i in range(len(self)):
            list_of_grps[y_mapping[y[i]]].add(self.list_of_nodes[i])
        return list_of_grps
    
    def cluster_grid(self, grid_size):
        coordinates = self.get_coords()
        y = np.zeros(coordinates.shape[0], dtype=np.int32)
        grid = Grid(coordinates, grid_size)
        list_of_indices = grid.group_points(coordinates)
        for y_i, list_i in enumerate(list_of_indices):
            y[list_i] = y_i
        return self.make_sub_grps(y)
        
    def cluster_dbscan(self, min_dst=16):
        cl = DBSCAN(eps=min_dst)
        y = cl.fit_predict(self.get_coords())
        return self.make_sub_grps(y)
    
    def cluster_kmeans(self, n_clusters=16):
        cl = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto")
        y = cl.fit_predict(self.get_coords())
        return self.make_sub_grps(y)
    
    def cluster_agglomerative(self, n_clusters=8, distance_threshold=None):
        cl = AgglomerativeClustering(n_clusters, distance_threshold=distance_threshold)
        y = cl.fit_predict(self.get_coords())
        return self.make_sub_grps(y)
