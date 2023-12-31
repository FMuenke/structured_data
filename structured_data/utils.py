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
