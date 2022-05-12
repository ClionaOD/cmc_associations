import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def hierarchical_clustering(in_matrix, label_list, outpath=None):
    """
    perform agglomerative hierarchical clustering with ward linkage on the provided distance matrix
    args:
        in_matrix : the distance matrix you wish to cluster, either condensed or not. 
                    It should be input as a numpy array.
        label_list : the list of labels corresponding to the rows/cols of the distance matrix.
        outpath : str, if provided the dendrogram will save to this path

    returns:
        cluster_order : the list of labels according to the clustering. Used to reorganise
                        an RDM according to cluster structure.
    """
    matrix = copy.copy(in_matrix)
    if matrix.ndim == 2:
        matrix = ssd.squareform(matrix)

    fig,ax = plt.subplots(figsize=(15,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='single'), 
        ax=ax, 
        labels=label_list, 
        orientation='right'
    )
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if outpath:
        plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order