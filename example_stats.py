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


def d_connected_triplet_motif_density(g):
    """
    Return the uncorrected density of connected triplet motifs. "Uncorrected" means this function counts subgraphs
    isomorphic to each three-node motif, not *induced* subgraphs. To "correct" the output, do the following:

    ```
    uncorrected_densities = d_connected_triplet_motif_density(g)
    corrected_densities = triplet_over_to_exact.dot(uncorrected_densities)
    ```

    :param g: Adjacency matrix (possibly sparse)
    :return: vector of 15 (uncorrected) motif counts of all nonempty triplet motifs
    """
    # TODO put some thought into efficiently allocating memory/matrix operations
    n = g.shape[0]
    counts = np.zeros(15)
    num_edges = g.sum()
    g_squared = g.dot(g)
    tr_g_sq = g_squared.diagonal().sum()  # works with sparse matrices, not with numpy arrays?

    counts[0] = (n - 2) * num_edges  # (over)counts triplets of nodes with at least 1 edge
    counts[1] = (n - 2) * tr_g_sq / 2  # (over)counts triplets of nodes with at least 1 recip
    counts[2] = (g_squared.sum() - tr_g_sq)  # sum of off-diagonal elements of A^2

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

    return 6 * counts / (n * (n - 1) * (n - 2))


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
    delta[0] = n - 2  # scale factor from definition above
    delta[1] = (n - 2) * g[v, u]

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

    return 6 * Duv * delta / (n * (n - 1) * (n - 2))


S_string = [
    "122223333444456",  # 0
    "010000011111223",  # 1
    "001001311232246",  # 2
    "000101010211123",  # 3
    "000011001112123",  # 4
    "000001000212036",  # 5
    "000000100010012",  # 6
    "000000010210236",  # 7
    "000000001012236",  # 8
    "000000000100013",  # 9
    "000000000010026",  # 10
    "000000000001013",  # 11
    "000000000000113",  # 12
    "000000000000016",  # 13
    "000000000000001"]  # 14
# automorphisms = [1, 2, 1, 2, 2, 3, 1, 1, 1, 2, 1, 2, 2, 1, 6]
# triplet_exact_to_over = np.array([[int(d) * automorphisms[i] for d in s] for i,s in enumerate(S_string)])
triplet_exact_to_over = np.array([[int(d) for d in s] for i, s in enumerate(S_string)])
# print(triplet_exact_to_over)
triplet_over_to_exact = np.linalg.inv(triplet_exact_to_over).astype(int)
# print(triplet_over_to_exact)


def d_connected_triplet_motif_density_dense(g):
    """
    Python is garbage, and should be thrown in the dumpster, because it is garbage.
    I am rewriting this function here to handle dense numpy arrays, since apparently they don't have a `.multiply`
    method for entrywise multiplication.

    Return the uncorrected density of connected triplet motifs. "Uncorrected" means this function counts subgraphs
    isomorphic to each three-node motif, not *induced* subgraphs. To "correct" the output, do the following:

    ```
    uncorrected_densities = d_connected_triplet_motif_density(g)
    corrected_densities = triplet_over_to_exact.dot(uncorrected_densities)
    ```

    :param g: Adjacency matrix (possibly sparse)
    :return: vector of 15 (uncorrected) motif counts of all nonempty triplet motifs
    """
    # TODO put some thought into efficiently allocating memory/matrix operations
    n = g.shape[0]
    counts = np.zeros(15)
    num_edges = g.sum()
    g_squared = g.dot(g)
    tr_g_sq = g_squared.diagonal().sum()  # works with sparse matrices, not with numpy arrays?

    counts[0] = (n - 2) * num_edges  # (over)counts triplets of nodes with at least 1 edge
    counts[1] = (n - 2) * tr_g_sq / 2  # (over)counts triplets of nodes with at least 1 recip
    counts[2] = (g_squared.sum() - tr_g_sq)

    diverging = (g.T).dot(g)  # ATA, common pre
    converging = g.dot(g.T)  # AAT, common post
    counts[3] = (diverging.sum() - num_edges) / 2
    counts[4] = (converging.sum() - num_edges) / 2
    counts[5] = np.multiply(g, g_squared).sum()

    g_cubed = g_squared.dot(g)
    counts[6] = g_cubed.diagonal().sum() / 3

    g_sym = np.multiply(g, g.T)
    bidegi = g_sym.sum(axis=1)[:, None]
    counts[7] = np.multiply(g, bidegi).sum() - g_sym.sum()
    counts[8] = np.multiply(g, bidegi.T).sum() - g_sym.sum()
    counts[9] = np.multiply(g_sym, converging).sum() / 2
    counts[10] = np.multiply(g_sym, g_squared).sum()
    counts[11] = np.multiply(g_sym, diverging).sum() / 2

    g_sym_squared = g_sym.dot(g_sym)
    counts[12] = (g_sym_squared.sum() - g_sym_squared.diagonal().sum()) / 2
    counts[13] = np.multiply(g_sym_squared, g).sum()
    counts[14] = g_sym_squared.dot(g_sym).diagonal().sum() / 6

    return 6 * counts / (n * (n - 1) * (n - 2))


def d_delta_connected_triplet_motif_density_dense(g, u, v):
    """
    Python is garbage, and should be thrown in the dumpster, because it is garbage.
    I am rewriting this function here to handle dense numpy arrays, since apparently they don't have a `.multiply`
    method for entrywise multiplication.

    Return the change in uncorrected motif density when edge `u,v` is toggled
    :param g: sparse adjacency matrix
    :param u: source of edge
    :param v: sink of edge
    :return: change in counts
    """
    n = g.shape[0]
    Duv = 2 * g[u, v] - 1
    delta = np.zeros(15)  # delta will get multiplied by Duv at the end
    delta[0] = n - 2  # scale factor from definition above
    delta[1] = (n - 2) * g[v, u]

    outu = g[u, :].sum()
    inu = g[:, u].sum()
    outv = g[v, :].sum()
    inv = g[:, v].sum()
    delta[2] = inu + outv - 2 * g[v, u]
    delta[3] = outu - g[u, v]
    delta[4] = inv - g[u, v]

    common_post = np.multiply(g[u, :], g[v, :])
    common_pre = np.multiply(g[:, u], g[:, v])
    delta[5] = common_post.sum() + common_pre.sum() + g[u, :].dot(g[:, v])
    delta[6] = g[v, :].dot(g[:, u])

    bidegu = g[u, :].dot(g[:, u])
    bidegv = g[v, :].dot(g[:, v])
    delta[7] = bidegu + g[v, u] * (outu + outv - 2 * g[u, v] - 2 * g[v, u] + 1)
    delta[8] = bidegv + g[v, u] * (inu + inv - 2 * g[u, v] - 2 * g[v, u] + 1)

    delta[9] = g[u, :].dot(common_pre) + g[v, u] * common_post.sum()  # additional term below if g[v,u] is not 0
    delta[10] = common_post.dot(g[:, u]) + g[v, :].dot(common_pre)  # additional stuff below if g[v,u]
    delta[11] = common_post.dot(g[:, v]) + g[v, u] * common_pre.sum()  # additional stuff if g[v,u] is not 0

    delta[12] = g[v, u] * (bidegu + bidegv - 2 * g[u, v])
    delta[13] = common_post.dot(common_pre)  # additional stuff if g[v,u] is not 0
    if g[v, u] != 0:
        delta[10] = delta[10] + g[u, :].dot(g[:, v]) + g[v, :].dot(g[:, u])
        delta[13] = delta[13] + common_post.dot(g[:, v]) + common_post.dot(g[:, u]) + \
            g[u, :].dot(common_pre) + g[v, :].dot(common_pre)
        delta[14] = common_post.dot(common_pre)

    return 6 * Duv * delta / (n * (n - 1) * (n - 2))
