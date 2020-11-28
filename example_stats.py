"""Implementations of example graph statistics and difference functions.

Function names that begin with `u_` or `d_` only work on undirected or directed graphs, respectively; if there is no
prefix the function will return the correct value in either case. """
import numpy as np
from scipy import sparse


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


def d_connected_triplet_motif_density(g):
    """
    Return the uncorrected density of connected triplet motifs.

    !!! Correction will be implemented eventually

    :param g: Adjacency matrix (possibly sparse)
    :return: vector of 15 (uncorrected) motif counts of all triplet motifs

    "Uncorrected" means, e.g. the first entry is the edge density, rather than the number of triplets with a single
    edge. This function counts subgraphs isomorphic to the 3-node motifs, not induced subgraphs.
    """
    # TODO put some thought into efficiently allocating memory/matrix operations
    n = g.shape[0]
    counts = np.zeros(15)
    num_edges = g.sum()
    g_squared = g.dot(g)
    tr_g_sq = g_squared.diagonal().sum()  # works with sparse matrices, not with numpy arrays?

    counts[0] = n * num_edges  # scale by n since everything is getting divided by n^3
    counts[1] = n * tr_g_sq / 2
    counts[2] = (g_squared.sum() - tr_g_sq)

    diverging = (g.T).dot(g)  # ATA, common pre
    converging = g.dot(g.T)  # AAT, common post
    counts[3] = (diverging.sum() - num_edges) / 2
    counts[4] = (converging.sum() - num_edges) / 2
    counts[5] = g.multiply(g_squared).sum()

    g_cubed = g_squared.dot(g)
    counts[6] = g_cubed.diagonal().sum() / 3

    g_sym = g.multiply(g.T)
    bidegi = g_sym.sum(axis=1)
    counts[7] = g.multiply(bidegi).sum() - g_sym.sum()
    counts[8] = g.multiply(bidegi.T).sum() - g_sym.sum()
    counts[9] = g_sym.multiply(converging).sum() / 2
    counts[10] = g_sym.multiply(g_squared).sum()
    counts[11] = g_sym.multiply(diverging).sum() / 2

    g_sym_squared = g_sym.dot(g_sym)
    counts[12] = (g_sym_squared.sum() - g_sym_squared.diagonal().sum()) / 2
    counts[13] = g_sym_squared.multiply(g).sum()
    counts[14] = g_sym_squared.dot(g_sym).diagonal().sum() / 6

    return counts / (n ** 3)


def d_delta_connected_triplet_motif_density(g, u, v):
    """
    Return the change in uncorrected motif density when edge `u,v` is toggled
    :param g: sparse adjacency matrix
    :param u: source of edge
    :param v: sink of edge
    :return: change in counts
    """
    n = g.shape[0]
    Duv = 2 * g[u, v] - 1
    delta = np.zeros(15)  # delta will get multiplied by Duv at the end
    delta[0] = n  # scale factor from definition above
    delta[1] = n * g[v, u]

    outu = g[u, :].sum()
    inu = g[:, u].sum()
    outv = g[v, :].sum()
    inv = g[:, v].sum()
    delta[2] = inu + outv - 2 * g[v, u]
    delta[3] = outu - g[u, v]
    delta[4] = inv - g[u, v]

    common_post = g[u, :].multiply(g[v, :])
    common_pre = g[:, u].multiply(g[:, v])
    delta[5] = common_post.sum() + common_pre.sum() + g[u, :].dot(g[:, v])[0, 0]
    delta[6] = g[v, :].dot(g[:, u])[0, 0]

    bidegu = g[u, :].dot(g[:, u])[0, 0]
    bidegv = g[v, :].dot(g[:, v])[0, 0]
    delta[7] = bidegu + g[v, u] * (outu + outv - 2 * g[u, v] - 2 * g[v, u] + 1)
    delta[8] = bidegv + g[v, u] * (inu + inv - 2 * g[u, v] - 2 * g[v, u] + 1)

    delta[9] = g[u, :].dot(common_pre)[0, 0] + g[v, u] * common_post.sum()  # additional term below if g[v,u] is not 0
    delta[10] = common_post.dot(g[:, u])[0, 0] + g[v, :].dot(common_pre)[0, 0]  # additional stuff below if g[v,u]
    delta[11] = common_post.dot(g[:, v])[0, 0] + g[v, u] * common_pre.sum()  # additional stuff if g[v,u] is not 0

    delta[12] = g[v, u] * (bidegu + bidegv - 2 * g[u, v])
    delta[13] = common_post.dot(common_pre)[0, 0]  # additional stuff if g[v,u] is not 0
    if g[v, u] != 0:
        delta[10] = delta[10] + g[u, :].dot(g[:, v])[0, 0] + g[v, :].dot(g[:, u])[0, 0]
        delta[13] = delta[13] + common_post.dot(g[:, v])[0, 0] + common_post.dot(g[:, u])[0, 0] + \
                    g[u, :].dot(common_pre)[0, 0] + g[v, :].dot(common_pre)[0, 0]
        delta[14] = common_post.dot(common_pre)[0, 0]

    return Duv * delta / (n ** 3)
