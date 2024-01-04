import numpy as np
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def shilouette_score_list_of_group_of_nodes(list_of_group_of_nodes):
    x, y = [], []
    for i, grp_of_nodes in enumerate(list_of_group_of_nodes):
        x_k = grp_of_nodes.get_x()
        y_k = i * np.ones(x_k.shape[0])
        x.append(x_k)
        y.append(y_k)
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    score = silhouette_score(x, y)
    return score


def dst(x, x_samples):
    return np.sqrt(np.sum(np.square(x - x_samples), axis=1))


def pairwise_distance_matrix(vectors1, vectors2):
    return cdist(vectors1, vectors2, 'euclidean')


def minimize_total_distance(vectors1, vectors2):
    distance_matrix = pairwise_distance_matrix(vectors1, vectors2)
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    total_distance = distance_matrix[row_indices, col_indices].sum()
    return row_indices, col_indices, total_distance


def match_grp_of_nodes(grp_1, grp_2):
    vectors_set1 = [node.representation for node in grp_1]
    vectors_set2 = [node.representation for node in grp_2]
    row_indices, col_indices, _ = minimize_total_distance(vectors_set1, vectors_set2)
    matches = [[grp_1[i].index, grp_2[j].index] for i, j in zip(row_indices, col_indices)]
    return matches


def group_points_by_grid(points, grid_size):
    """
    Downsample points by summarizing all points within each grid cell.

    Parameters:
    - points: 2D array-like, shape (n_samples, coordinates)
        The input points.
    - grid_size: float or tuple
        If a float, it represents the side length of a square grid cell.
        If a tuple (x_size, y_size), it represents the side lengths of a rectangular grid cell.

    Returns:
    - list_of_indices: 2D array
        all indices of points with in the corresponding grid cell.
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

    return list_of_indices, grid_centers
