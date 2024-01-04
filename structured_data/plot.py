import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging


def plot_multiple_group_of_nodes(list_of_group_of_nodes, labels=None):
    x, y = [], []
    for i, grp_of_nodes in enumerate(list_of_group_of_nodes):
        if len(grp_of_nodes) == 0:
            logging.warning("Empty Group of Nodes")
            continue
        x.append(grp_of_nodes.get_x())
        if labels is None:
            lab = i
        else:
            lab = labels[i]
        for _ in grp_of_nodes:
            y.append(lab)

    x = np.concatenate(x, axis=0)
    if x.shape[1] > 2:
        red = UMAP(n_components=2)
        x = red.fit_transform(x)
        
    df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})
    sns.scatterplot(data=df, x="x1", y="x2", hue="y")
    plt.show()

def plot_temporal_group_of_nodes(list_of_temporal_nodes):
    x, y = [], []
    for node in list_of_temporal_nodes:
        x.append(node.to_datetime())
        y.append(node.get_repr())

    y = np.array(y)
    if len(y.shape) > 1:
        red = PCA(n_components=1)
        y = red.fit_transform(y)
    df = pd.DataFrame({"time": x, "y": y})
    sns.scatterplot(data=df, x="time", y="y")
    plt.show()

def plot_spatial_group_of_nodes(list_of_spatial_nodes):
    x, y = [], []
    for node in list_of_spatial_nodes:
        x.append(node.representation)
        y.append(node.get_repr())

    x, y = np.array(x), np.array(y)
    if len(y.shape) > 1:
        red = PCA(n_components=1)
        y = red.fit_transform(y)
    df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})
    sns.scatterplot(data=df, x="x1", y="x2", hue="y")
    plt.show()


def plot_hist(list_of_nodes, labels):
    df = {"feature": [], "value": []}
    if type(labels) is not list:
        labels = [labels]
    for node in list_of_nodes:
        repr = node.get_repr()
        if type(repr) is not list:
            repr = [repr]
        for i, lab in enumerate(labels):
            df["feature"].append(lab)
            df["value"].append(repr[i])

    sns.histplot(data=df, x="value", hue="feature")
    plt.show()