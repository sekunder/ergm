"""Implementations of example graph statistics and difference functions.

Function names that begin with `u_` or `d_` only work on undirected or directed graphs, respectively; if there is no
prefix the function will return the correct value in either case. """
import numpy as np


def u_num_edges(g):
    """Return the number of edges in undirected graph `g`"""
    return np.array([g.sum() / 2])


def u_delta_num_edges_undirected(g, u, v):
    """Return the change in the number of edges if edge `u,v` is toggled"""
    # if u,v is an edge, then toggling it will yield (old - new) = 1
    # otherwise, toggling it yields (old - new) = -1
    return 2 * g[u, v] - 1


def u_edge_triangle_density(g):
    """Return the edge density and triangle density of `g`"""
    n = g.shape[0]
    return np.array([g.sum() / (n * (n - 1)), np.trace(np.matmul(np.matmul(g, g), g)) / (n * (n - 1) * (n - 2))])


def u_delta_edge_triangle_density(g, u, v):
    """Return the change in edge density and triangle density when edge `u,v` is toggled"""
    n = g.shape[0]
    delta_uv = 2 * g[u, v] - 1
    return np.array([2 * delta_uv / (n * (n - 1)), 6 * delta_uv * np.dot(g[u], g[v]) / (n * (n - 1) * (n - 2))])
