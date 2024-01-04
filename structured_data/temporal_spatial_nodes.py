import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
from structured_data.spatial_nodes import SpatialNode, GroupOfSpatialNodes

class TemporalSpatialNode(SpatialNode):
    def __init__(self, index, time_stamp, coordinates, representation, sample=None):
        self.time_stamp = int(time_stamp)
        super().__init__(index, coordinates, representation, sample)

    def to_spatial_node(self):
        return SpatialNode(self.index, self.x, self.sample)
    
    def to_datetime(self):
        return datetime.fromtimestamp(self.time_stamp)
    


class GroupOfTemporalSpatialNodes(GroupOfSpatialNodes):
    def __init__(self, list_of_temporal_spatial_nodes):
        super().__init__(list_of_temporal_spatial_nodes)

    def to_spatial_nodes(self):
        return GroupOfSpatialNodes([node.to_spatial_node() for node in self.list_of_nodes])
    
    def summarize(self, summary_func=np.mean):
        sum_index = self.list_of_nodes[0].index
        sum_time = np.mean(self.get_time())
        coordinate = np.mean(np.array([node.representation for node in self.list_of_nodes]), axis=0)
        representation = summary_func(np.array([node.get_repr() for node in self.list_of_nodes]), axis=0)
        sum_sample = self.list_of_nodes[0].sample
        return TemporalSpatialNode(sum_index, sum_time, coordinate, representation, sum_sample)
    
    def get_time(self):
        return np.array([node.time_stamp for node in self.list_of_nodes]).reshape(-1, 1)
    
    def make_sub_grps(self, y):
        assert len(y) == len(self), "Assignement Y does not match number of nodes"
        list_of_grps = [GroupOfTemporalSpatialNodes([]) for _ in np.unique(y)]
        for i in range(len(self)):
            list_of_grps[int(y[i])].add(self.list_of_nodes[i])
        return list_of_grps
    
    def split_by_time(self, delta_time):
        cl = DBSCAN(eps=delta_time)
        y = cl.fit_predict(self.get_time())
        return self.make_sub_grps(y)
    
    def depth(self, delta_time):
        return len(self.split_by_time(delta_time))
    
    def downsample_by_time(self, delta_time, summary_func=np.mean):
        sub_grps = self.split_by_time(delta_time)
        return GroupOfTemporalSpatialNodes([grp.summarize(summary_func) for grp in sub_grps])
    
    def compute_change(self, delta_time):
        summarized_grp = self.downsample_by_time(delta_time)
        repr = summarized_grp.get_repr()
        max_deviation = np.max(repr, axis=0) - np.min(repr, axis=0)
        return max_deviation
