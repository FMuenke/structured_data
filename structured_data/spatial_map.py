import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from structured_data.spatial_nodes import SpatialNode, GroupOfSpatialNodes


def downsample_grid(points, grid_size, summary_func=np.mean):
    """
    Downsample points by summarizing all points within each grid cell.

    Parameters:
    - points: 2D array-like, shape (n_samples, n_features)
        The input points.
    - grid_size: float or tuple
        If a float, it represents the side length of a square grid cell.
        If a tuple (x_size, y_size), it represents the side lengths of a rectangular grid cell.
    - summary_func: function, optional
        The function used to summarize points within each grid cell.
        Default is np.mean.

    Returns:
    - downsampled_points: 2D array
        The downsampled points representing the summaries within each grid cell.
    - grid_centers: 2D array
        The coordinates of the grid cell centers.
    """

    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    x_bins = np.arange(min_x, max_x + grid_size[0], grid_size[0])
    y_bins = np.arange(min_y, max_y + grid_size[1], grid_size[1])

    digitized_x = np.digitize(points[:, 0], x_bins)
    digitized_y = np.digitize(points[:, 1], y_bins)

    grid_centers = np.array([[x_bins[i] + grid_size[0] / 2, y_bins[j] + grid_size[1] / 2]
                             for i in range(1, len(x_bins)) for j in range(1, len(y_bins))])
    
    list_of_indices = [
        np.where((digitized_x == i) & (digitized_y == j))[0].tolist()
        for i in range(1, len(x_bins)) for j in range(1, len(y_bins))
    ]

    downsampled_points = np.array([[summary_func(points[(digitized_x == i) & (digitized_y == j)][:, 0]),
                                    summary_func(points[(digitized_x == i) & (digitized_y == j)][:, 1])]
                                   for i in range(1, len(x_bins)) for j in range(1, len(y_bins))])

    return downsampled_points, list_of_indices, grid_centers


class SpatialMap:
    def __init__(self, grp_of_spatial_nodes):
        self.n = len(grp_of_spatial_nodes)
        self.list_of_nodes = grp_of_spatial_nodes
        self.coordinates = grp_of_spatial_nodes.get_x()
        self.min = np.min(self.coordinates, axis=0)
        self.max = np.max(self.coordinates, axis=0)
        
    def __len__(self):
        return self.n
    
    def __iter__(self):
        return self.list_of_nodes.__iter__()
    
    def get_x(self):
        return self.coordinates
    
    def query_by_coords(self, coords, n, dst_min=None):
        distances = np.sqrt(np.sum(np.square(self.coordinates - coords), axis=1))
        if dst_min is not None:
            distances[distances > dst_min] = np.inf
        sorted_indices = np.argsort(distances)
        geo_nodes = [self.list_of_nodes[i] for i in sorted_indices[:n] if distances[i] < np.inf]
        return GroupOfSpatialNodes(geo_nodes)

    def cut(self, coords, radius):
        grp = self.query_by_coords(coords, self.n, radius)
        return SpatialMap(grp)
    
    def downsample_by_grid(self, grid_size):
        _, list_of_indices, _ = downsample_grid(self.coordinates, grid_size)
        new_nodes = GroupOfSpatialNodes([])
        for list_i in list_of_indices:
            if len(list_i) == 0:
                continue
            grp = GroupOfSpatialNodes([self.list_of_nodes[i] for i in list_i])
            new_nodes.add(grp.summarize())
        return SpatialMap(new_nodes)

    def get_coverage(self, grid_size=4):
        _, list_of_indices, _ = downsample_grid(self.coordinates, grid_size)
        list_of_active_indices = [list_i for list_i in list_of_indices if len(list_i) > 0] 
        return len(list_of_active_indices)
    
    def make_sub_maps(self, y):
        assert len(y) == self.n, "Assignement Y does not match number of nodes"
        list_of_grps = [GroupOfSpatialNodes([]) for _ in np.unique(y)]
        for i in range(self.n):
            list_of_grps[y[i]].add(self.list_of_nodes[i])
        return [SpatialMap(grp) for grp in list_of_grps]
    
    def cluster_dbscan(self, min_dst=16):
        cl = DBSCAN(eps=min_dst)
        y = cl.fit_predict(self.coordinates)
        return self.make_sub_maps(y)
    
    def cluster_spectral(self, n_clusters=8):
        cl = SpectralClustering(n_clusters)
        y = cl.fit_predict(self.coordinates)
        return self.make_sub_maps(y)
    
    def cluster_agglomerative(self, n_clusters=8, distance_threshold=None):
        cl = AgglomerativeClustering(n_clusters, distance_threshold=distance_threshold)
        y = cl.fit_predict(self.coordinates)
        return self.make_sub_maps(y)

    def plot(self, min_dst):
        cl = DBSCAN(eps=min_dst, min_samples=1)
        y = cl.fit_predict(self.coordinates)
        df = pd.DataFrame({"x1": self.coordinates[:, 0], "x2": self.coordinates[:, 1], "y": y})
        sns.scatterplot(data=df, x="x1", y="x2", hue=y)
        plt.show()
