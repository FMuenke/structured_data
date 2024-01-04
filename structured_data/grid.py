import numpy as np


class Grid:
    def __init__(self, points, grid_size):

        if not isinstance(grid_size, tuple):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size

        self.min_x, self.min_y = np.min(points, axis=0)
        self.max_x, self.max_y = np.max(points, axis=0)

        self.x_bins = np.arange(self.min_x, self.max_x + grid_size[0], grid_size[0])
        self.y_bins = np.arange(self.min_y, self.max_y + grid_size[1], grid_size[1])

        grid_centers = np.array([[self.x_bins[i] + grid_size[0] / 2, self.y_bins[j] + grid_size[1] / 2]
                                for i in range(1, len(self.x_bins)) for j in range(1, len(self.y_bins))])
        self.grid_centers = grid_centers

    def group_points(self, points, return_only_active=True):
        digitized_x = np.digitize(points[:, 0], self.x_bins)
        digitized_y = np.digitize(points[:, 1], self.y_bins)

        list_of_indices = [
            np.where((digitized_x == i) & (digitized_y == j))[0].tolist()
            for i in range(1, len(self.x_bins)) for j in range(1, len(self.y_bins))
        ]

        if return_only_active:
            return [list_i for list_i in list_of_indices if len(list_i) > 0]
        return list_of_indices
    
    def find_common_cells(self, multiple_points):
        n_point_sets = len(multiple_points)
        grid_occupation = np.zeros((len(self.grid_centers), n_point_sets))
        grouped_points = [self.group_points(points, return_only_active=False) for points in multiple_points]

        for i, _ in enumerate(multiple_points):
            list_of_indices = grouped_points[i]
            for grid_i, list_i in enumerate(list_of_indices):
                if len(list_i) == 0:
                    continue
                grid_occupation[grid_i, i] = 1
        
        grid_occupation = np.sum(grid_occupation, axis=1)
        common_groups = [[] * n_point_sets]
        for grid_i, grid_oc in enumerate(grid_occupation):
            if grid_oc < n_point_sets:
                continue
            for i, list_of_indices in enumerate(grouped_points):
                common_groups[i].append(list_of_indices[grid_i])
        return common_groups
        