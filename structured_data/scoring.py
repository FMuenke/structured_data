import numpy as np


def euclid_d(x_t, x_B):
    return np.sqrt(np.sum(np.square(x_t - x_B), axis=1))


def mse_d(x_t, x_B):
    return np.mean(np.square(x_t - x_B), axis=1)


def mae_d(x_t, x_B):
    return np.mean(np.abs(x_t - x_B), axis=1)


def manhatten_d(x_t, x_B):
    return np.sum(np.abs(x_t - x_B), axis=1)


def consine_d(x_t, x_B):
    norm_x = np.linalg.norm(x_t) * np.linalg.norm(x_B, axis=1)
    norm_x[norm_x == 0] = 1e-6
    return np.dot(x_t, x_B.T)/(norm_x)


def compute_representation_distance(representations, func=mae_d):
    distances = []
    for i in range(len(representations)):
        x_t = representations[i]
        x_B = np.array(representations[:i] + representations[i:])
        dst = func(x_t, x_B)
        distances.append(np.mean(dst))
    return distances


def compute_intra_object_deviation(list_of_group_of_nodes, delta_time=0, agg_func=np.mean):
    robustness = []
    for gon in list_of_group_of_nodes:
        if delta_time == 0:
            aggregated_representations = [node.get_repr() for node in gon]
        else:
            split_grps = gon.split_by_time(delta_time=delta_time)
            aggregated_representations = [grp.aggregate(agg_func) for grp in split_grps]
        distances = compute_representation_distance(aggregated_representations)
        score = np.mean(distances)
        robustness.append(score)
    return robustness


def compute_inter_object_deviation(list_of_group_of_nodes, agg_func=np.mean):
    aggregated_representations = [gon.aggregate(agg_func) for gon in list_of_group_of_nodes]
    descriptiveness = compute_representation_distance(aggregated_representations)
    return descriptiveness