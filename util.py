"""
utility functions for ergm
"""
import numpy as np
import datetime
import sys

import networkx as nx


# from scipy import sparse

def log_msg(*args, out=sys.stdout, **kwargs):
    """Print message m with a timestamp if out is not None."""
    if out:
        print(datetime.datetime.now().strftime("%Y %m %d %H:%M:%S "), *args, **kwargs, file=out)


def index_to_edge(idx, n, directed=True, order="columns"):
    """
    Returns the ordered pair `(e0,e1)` for the edge which has linear index `idx`. This is essentially the linear
    index of an entry in a matrix, except shifts are included so the diagonal entries don't get indexed.

    :param idx: an integer between 0 and n*(n-1) (inclusive) for directed graphs, or 0 and n*(n-1)/2 for undirected.
    :param n: the number of nodes in the graph
    :param directed: whether to find the index for a directed (all off-diagonal entries used) or undirected
                   (upper triangle only). Default: true
    :param order: Whether matrix entries are indexed in column order or row order. Default columns, so 0 maps to (1,0),
                and so on down the 0th column before moving to the 1th column. Options are "columns" (default)
                or "rows".
    :return: tuple of integers, the indices in the adjacency matrix.
    """

    if directed:
        e1 = idx // (n - 1)
        e0 = idx % (n - 1) + (idx % (n - 1) >= e1)
    else:
        e1 = np.ceil(triangular_root(idx + 1)).astype(int)
        e0 = idx - (e1 - 1) * e1 // 2

    if order == "columns":
        return np.array([e0, e1])
    else:
        return np.array([e1, e0])


def triangular_root(x):
    """Returns the triangular root of x. If this returns an integer, x is a triangular number; otherwise, it lies between two triangular numbers.

    See https://en.wikipedia.org/wiki/Triangular_number"""
    return (np.sqrt(8 * x + 1) - 1) / 2


def triangular_number(n):
    """Returns the `n`th triangular number `n * (n + 1) // 2`"""
    return n * (n + 1) // 2


def ellipse(center, v1, v2, resolution=10):
    """
    Returns two arrays, `x`, `y`, such that `plt.plot(x,y)` will be an ellipse with specified center and axes.
    The `resolution` parameter adjusts the number of points around the ellipse
    :param center: coordinates of center
    :param v1: 1st axis
    :param v2: 2nd axis
    :param resolution: number of samples points around ellipse
    :return: Two arrays
    """
    ls = np.linspace(0, 2 * np.pi, num=resolution)
    x = center[0] + np.cos(ls) * v1[0] + np.sin(ls) * v2[0]
    y = center[1] + np.cos(ls) * v1[1] + np.sin(ls) * v2[1]
    return x, y


def networkx_graph_to_sparse_array(g):
    """
    Convert networkx graph g into a binary matrix with 0s on the diagonal. Entry `i,j` indicates whether there is an
    edge from node `i` to node `j`. Indices are the order in which nodes are returned from `g.nodes`.

    This will generate a warning as it involves manually setting the diagonal to 0, which changes the sparsity
    structure of the matrix.

    :param g: `networkx` graph
    :return: a CSR sparse matrix
    """
    sg = nx.convert_matrix.to_scipy_sparse_matrix(g, dtype=int, weight=None)
    sg.data = np.ones_like(sg.data)
    sg.setdiag(sg.diagonal() * 0)
    return sg


def flatten(t):
    return [item for sublist in t for item in sublist]