"""Matrix algebra for counting triplet motifs"""
import numpy as np
from itertools import permutations

from util import index_to_directed_triplet_motif_matrix, basis_vector, bin_subsets, binary_digits


def tupp(m):
    return tuple(m.ravel())


def matt(m):
    return np.array(m).reshape(3, 3)


################################################################################
# Isomorphisms of 3 node graphs
################################################################################
permutation_matrices = {(s0, s1, s2): np.array([basis_vector(s0), basis_vector(s1), basis_vector(s2)]) for s0, s1, s2 in permutations(range(3))}


def get_permutation_matrices():
    """
    There's some aspects of python's import system that I cannot seem to figure
    out, so there's going to be a lot of these `get_blah()` functions in this
    file.
    """
    return permutation_matrices


def isomorphic(tg, th):
    """Check if graphs, represented by raveled tuples tg, th, are isomorphic"""
    mg = matt(tg)
    mh = matt(th)
    for P in permutation_matrices.values():
        if np.all(P @ mg @ P.T == mh):
            return True
    return False


def color_isomorphic(tg, th, cg, ch):
    mg = matt(tg)
    mh = matt(th)
    for s, P in permutation_matrices.items():
        colors_match = all(cg[si] == ch[i] for si, i in zip(s, range(3)))
        if colors_match and np.all(P @ mg @ P.T == mh):
            return True
    return False

################################################################################
# Mapping from graphs to their isomorphism class
################################################################################


iso_representative_AAA = {}  # map from tuple to represantive tuple
iso_classes_AAA = {}  # preimage map from represntative tuple to list of tuples
for k in range(64):
    m = index_to_directed_triplet_motif_matrix(k)
    tm = tuple(m.ravel())  # current tuple
    for iso in iso_classes_AAA:  # iso is the representative tuple
        if isomorphic(tm, iso):
            iso_classes_AAA[iso].append(tm)
            iso_representative_AAA[tm] = iso
            break
    else:
        iso_classes_AAA[tm] = [tm]
        iso_representative_AAA[tm] = tm

iso_classes_AAB = {}  # preimage map from representative to list of tuples. Color is always AAB pattern.
iso_representative_AAB = {}  # map from tuple to representative tuple. Color is always AAB pattern.
colors_AAB = [0, 0, 1]
for k in range(64):
    # base graph has colors AAB
    m = index_to_directed_triplet_motif_matrix(k)
    tm = tuple(m.ravel())
    for iso in iso_classes_AAB:
        if color_isomorphic(tm, iso, colors_AAB, colors_AAB):
            iso_classes_AAB[iso].append(tm)
            iso_representative_AAB[tm] = iso
            break
    else:
        iso_classes_AAB[tm] = [tm]
        iso_representative_AAB[tm] = tm

iso_classes_ABB = {}  # preimage map from representative to list of tuples. Color is always ABB
iso_representative_ABB = {}  # map from tuple to representative tuple
colors_ABB = [0, 0, 1]
for k in range(64):
    # base graph has colors ABB
    m = index_to_directed_triplet_motif_matrix(k)
    tm = tuple(m.ravel())
    for iso in iso_classes_ABB:
        if color_isomorphic(tm, iso, colors_ABB, colors_ABB):
            iso_classes_ABB[iso].append(tm)
            iso_representative_ABB[tm] = iso
            break
    else:
        iso_classes_ABB[tm] = [tm]
        iso_representative_ABB[tm] = tm

iso_classes_ABC, iso_representative_ABC = {}, {}
for k in range(64):
    t = tupp(index_to_directed_triplet_motif_matrix(k))
    iso_classes_ABC[t] = [t]  # preimage map from representative to list of tuples. Always singleton
    iso_representative_ABC[t] = t  # map from tuple to representative tuple (just identity map)


def get_iso_classes(pat):
    if pat == "AAA":
        return iso_classes_AAA
    if pat == "AAB":
        return iso_classes_AAB
    if pat == "ABB":
        return iso_classes_ABB
    if pat == "ABC":
        return iso_classes_ABC
    return {}


def get_iso_representative(pat):
    if pat == "AAA":
        return iso_representative_AAA
    if pat == "AAB":
        return iso_representative_AAB
    if pat == "ABB":
        return iso_representative_ABB
    if pat == "ABC":
        return iso_representative_ABC
    return {}


def get_iso_list(pat):
    if pat == "AAA":
        return iso_list_AAA
    if pat == "AAB":
        return iso_list_AAB
    if pat == "ABB":
        return iso_list_ABB
    if pat == "ABC":
        return iso_list_ABC
    return []


def class_representative(ct, gt):
    """
    Get the graph tuple of the represntative of the class in whic `gt` lives
    given the nodes are colored according to the pattern in `ct`.
    Note that this skips pattern 'ABA' since all my current algorithms
    will avoid using that pattern.
    """
    if ct[0] == ct[1] == ct[2]:  # Case AAA
        return iso_representative_AAA[gt]
    if ct[0] == ct[1] != ct[2]:  # Case AAB
        return iso_representative_AAB[gt]
    if ct[0] != ct[1] == ct[2]:  # Case ABB
        return iso_representative_ABB[gt]
    if ct[0] != ct[1] != ct[2]:  # Case ABC
        return iso_representative_ABC[gt]
    return None

################################################################################
# Mobius Inversion-type matrices
################################################################################


iso_list_AAA = list(iso_classes_AAA.keys())
exact_to_over_AAA = np.zeros((len(iso_list_AAA), len(iso_list_AAA)), dtype=int)
for j, tm_j in enumerate(iso_list_AAA):
    for tm_i in bin_subsets(tm_j):
        i = iso_list_AAA.index(iso_representative_AAA[tm_i])
        exact_to_over_AAA[i, j] += 1
over_to_exact_AAA = np.linalg.inv(exact_to_over_AAA).astype(int)


iso_list_AAB = list(iso_classes_AAB.keys())
exact_to_over_AAB = np.zeros((len(iso_list_AAB), len(iso_list_AAB)), dtype=int)
for j, tm_j in enumerate(iso_list_AAB):
    for tm_i in bin_subsets(tm_j):
        iso_rep = iso_representative_AAB[tm_i]
        i = iso_list_AAB.index(iso_rep)
        exact_to_over_AAB[i, j] += 1
over_to_exact_AAB = np.linalg.inv(exact_to_over_AAB).astype(int)


iso_list_ABB = list(iso_classes_ABB.keys())
exact_to_over_ABB = np.zeros((len(iso_list_ABB), len(iso_list_ABB)), dtype=int)
for j, tm_j in enumerate(iso_list_ABB):
    for tm_i in bin_subsets(tm_j):
        iso_rep = iso_representative_ABB[tm_i]
        i = iso_list_ABB.index(iso_rep)
        exact_to_over_ABB[i, j] += 1
over_to_exact_ABB = np.linalg.inv(exact_to_over_ABB).astype(int)

iso_list_ABC = list(iso_classes_ABC.keys())
digits = binary_digits(np.arange(64), 6)
exact_to_over_ABC = np.zeros((64, 64), dtype=int)
for j in range(64):
    for t_i in bin_subsets(digits[j]):
        i = np.dot(t_i, [2 ** i for i in range(6)])
        exact_to_over_ABC[i, j] += 1
over_to_exact_ABC = np.linalg.inv(exact_to_over_ABC).astype(int)

################################################################################
# THE STAR OF THE SHOW
################################################################################


def overcount_three_color(c0, c1, c2, n0, n1, n2, sub01, sub02, sub10, sub12, sub20, sub21, verbose=False):
    """
    args:
    c0,c1,c2 : color names (will be checked against each other for correction factors)
    n0,n1,n1 : number of nodes of each color
    sub01..sub21 : the actual matrices of connections between colors

    Nodes are always ordered as follows:
      1
     / \
    0---2

    Returns overcounts, correction_matrix. to get exact counts do correction_matrix.dot(overcount)
    """
    if c0 == c1 == c2:
        # Color pattern AAA
        if verbose:
            print("Counting with pattern AAA")
        return overcount_AAA(sub01), over_to_exact_AAA
    if c0 == c1:
        if verbose:
            print("Counting with pattern AAB")
        # Color pattern AAB
        return overcount_AAB(n0, n2, sub01, sub02, sub20), over_to_exact_AAB
    if c1 == c2:
        if verbose:
            print("Counting with pattern ABB")
        # color pattern ABB
        return overcount_ABB(n0, n1, sub01, sub10, sub12), over_to_exact_ABB
    if verbose:
        print("Counting with pattern ABC")
    return overcount_ABC(n0, n1, n2, sub01, sub02, sub10, sub12, sub20, sub21), over_to_exact_ABC


def tr(M):
    return M.diagonal().sum()


def overcount_AAA(g):
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
    n = g.shape[0]
    counts = np.zeros(16, dtype=int)
    num_edges = g.sum()
    g_squared = g.dot(g)
    tr_g_sq = g_squared.diagonal().sum()  # works with sparse matrices, not with numpy arrays?

    counts[0] = n * (n - 1) * (n - 2) / 6  # count the number of triplets of nodes
    counts[1] = (n - 2) * num_edges  # (over)counts triplets of nodes with at least 1 edge
    counts[2] = (n - 2) * tr_g_sq / 2  # (over)counts triplets of nodes with at least 1 recip
    counts[4] = (g_squared.sum() - tr_g_sq)  # sum of off-diagonal elements of A^2

    diverging = (g.T).dot(g)  # ATA, common pre
    converging = g.dot(g.T)  # AAT, common post
    counts[3] = (diverging.sum() - num_edges) / 2
    counts[6] = (converging.sum() - num_edges) / 2
    counts[9] = g.multiply(g_squared).sum()

    g_cubed = g_squared.dot(g)
    counts[11] = g_cubed.diagonal().sum() / 3

    g_sym = g.multiply(g.T)
    bidegi = g_sym.sum(axis=1)
    counts[5] = g.multiply(bidegi).sum() - g_sym.sum()
    counts[7] = g.multiply(bidegi.T).sum() - g_sym.sum()
    counts[10] = g_sym.multiply(converging).sum() / 2
    counts[12] = g_sym.multiply(g_squared).sum()
    counts[13] = g_sym.multiply(diverging).sum() / 2

    g_sym_squared = g_sym.dot(g_sym)
    counts[8] = (g_sym_squared.sum() - g_sym_squared.diagonal().sum()) / 2
    counts[14] = g_sym_squared.multiply(g).sum()
    counts[15] = g_sym_squared.dot(g_sym).diagonal().sum() / 6

    return counts


def overcount_AAB(nA, nB, AA, AB, BA):
    counts = np.zeros(36, dtype=int)

    Bi_AA = AA.multiply(AA.T)
    Bi_AB = AB.multiply(BA.T)
#     Bi_BA = BA.multiply(AB.T)
    Bi_BA = Bi_AB.T
    counts[0] = (nA * (nA - 1) / 2) * nB  # disconnected
    counts[1] = AA.sum() * nB  # single AA edge
    counts[2] = Bi_AA.sum() * nB / 2  # single bidirectional AA edge
    counts[3] = (nA - 1) * AB.sum()  # single AB edge
    counts[4] = AA.T.dot(AB).sum()  # A <- A -> B
    counts[5] = AA.dot(AB).sum()  # A -> A -> B
    counts[6] = Bi_AA.dot(AB).sum()  # A <> A -> B
    counts[7] = (nA - 1) * BA.sum()  # single B to A edge
    counts[8] = BA.dot(AA).sum()  # B -> A -> A
    counts[9] = AA.dot(BA.T).sum()  # A -> A <- B
    counts[10] = Bi_AA.dot(BA.T).sum()  # A <> A <- B
    counts[11] = (nA - 1) * Bi_AB.sum()  # single A <> B
    counts[12] = Bi_BA.dot(AA).sum()  # B <> A -> A
    counts[13] = AA.dot(Bi_AB).sum()  # B <> A <- A
    counts[14] = Bi_AA.dot(Bi_AB).sum()  # A <> A <> B
    counts[15] = (AB.dot(AB.T).sum() - AB.dot(AB.T).diagonal().sum()) / 2
    counts[16] = AA.dot(AB).multiply(AB).sum()
    counts[17] = AB.dot(AB.T).multiply(Bi_AA).sum() / 2
    counts[18] = AB.dot(BA).sum() - AB.dot(BA).diagonal().sum()
    counts[19] = AA.dot(AB).dot(BA).diagonal().sum()  # directed cycle, may need to divide by 2 or 3 or 6
    counts[20] = AB.dot(BA).multiply(AA).sum()
    counts[21] = AB.dot(BA).multiply(Bi_AA).sum()
    counts[22] = AB.dot(Bi_BA).sum() - AB.dot(Bi_BA).diagonal().sum()
    counts[23] = AA.dot(AB).multiply(Bi_AB).sum()
    counts[24] = AA.T.dot(AB).multiply(Bi_AB).sum()
    counts[25] = Bi_AA.dot(Bi_AB).multiply(AB).sum()
    counts[26] = (BA.T.dot(BA).sum() - BA.T.dot(BA).diagonal().sum()) / 2
    counts[27] = BA.T.dot(BA).multiply(AA).sum()
    counts[28] = BA.T.dot(BA).multiply(Bi_AA).sum() / 2
    counts[29] = Bi_AB.dot(BA).sum() - Bi_AB.dot(BA).diagonal().sum()
    counts[30] = Bi_AB.dot(BA).multiply(AA).sum()
    counts[31] = BA.dot(AA).multiply(Bi_BA).sum()
    counts[32] = Bi_BA.dot(Bi_AA).multiply(BA).sum()
    counts[33] = (Bi_AB.dot(Bi_BA).sum() - Bi_AB.dot(Bi_BA).diagonal().sum()) / 2
    counts[34] = Bi_AB.dot(Bi_BA).multiply(AA).sum()
    counts[35] = Bi_AB.dot(Bi_BA).multiply(Bi_AA).sum() / 2

    return counts


def overcount_ABB(nA, nB, AB, BA, BB):
    # yes, it's a little silly to hand-code all these rather than doing some fancy isomorphism checks but this requires less thinking and fewer failure points.
    counts = np.zeros(36, dtype=int)

    Bi_AB = AB.multiply(BA.T)
    Bi_BA = Bi_AB.T
    Bi_BB = BB.multiply(BB.T)

    counts[0] = nA * (nB * (nB - 1) / 2)
    counts[1] = AB.sum() * (nB - 1)
    counts[2] = BA.sum() * (nB - 1)
    counts[3] = Bi_AB.sum() * (nB - 1)
    counts[4] = (AB.T.dot(AB).sum() - tr(AB.T.dot(AB))) / 2
    counts[5] = BA.dot(AB).sum() - tr(BA.dot(AB))
    counts[6] = Bi_BA.dot(AB).sum() - tr(Bi_BA.dot(AB))
    counts[7] = (BA.dot(BA.T).sum() - tr(BA.dot(BA.T))) / 2
    counts[8] = BA.dot(Bi_AB).sum() - tr(BA.dot(Bi_AB))
    counts[9] = (Bi_BA.dot(Bi_AB).sum() - tr(Bi_BA.dot(Bi_AB))) / 2
    counts[10] = BB.sum() * nA
    counts[11] = AB.dot(BB).sum()
    counts[12] = BA.T.dot(BB).sum()
    counts[13] = Bi_AB.dot(BB).sum()
    counts[14] = BB.dot(AB.T).sum()
    counts[15] = AB.dot(BB).multiply(AB).sum()
    counts[16] = BA.dot(AB).multiply(BB).sum()
    counts[17] = Bi_AB.dot(BB).multiply(AB).sum()
    counts[18] = BB.dot(BA).sum()
    counts[19] = tr(AB.dot(BB).dot(BA))
    counts[20] = BB.dot(BA).multiply(BA).sum()
    counts[21] = BB.dot(BA).multiply(Bi_BA).sum()
    counts[22] = BB.dot(Bi_BA).sum()
    counts[23] = AB.dot(BB).multiply(Bi_AB).sum()
    counts[24] = BB.dot(Bi_BA).multiply(BA).sum()
    counts[25] = Bi_BA.dot(Bi_AB).multiply(BB).sum()
    counts[26] = Bi_BB.sum() * nA / 2
    counts[27] = AB.dot(Bi_BB).sum()
    counts[28] = Bi_BB.dot(BA).sum()
    counts[29] = Bi_BB.dot(Bi_BA).sum()
    counts[30] = AB.T.dot(AB).multiply(Bi_BB).sum() / 2
    counts[31] = BA.dot(AB).multiply(Bi_BB).sum()
    counts[32] = Bi_AB.dot(Bi_BB).multiply(AB).sum()
    counts[33] = BA.dot(BA.T).multiply(Bi_BB).sum() / 2
    counts[34] = Bi_BB.dot(Bi_BA).multiply(BA).sum()
    counts[35] = Bi_BA.dot(Bi_AB).multiply(Bi_BB).sum() / 2

    return counts


def overcount_ABC(nA, nB, nC, AB, AC, BA, BC, CA, CB):
    counts = np.zeros(64, dtype=int)

    Bi_AB = AB.multiply(BA.T)
    Bi_BA = Bi_AB.T
    Bi_AC = AC.multiply(CA.T)
    Bi_CA = Bi_AC.T
    Bi_BC = BC.multiply(CB.T)
    Bi_CB = Bi_BC.T

    counts[0] = nA * nB * nC
    counts[1] = AB.sum() * nC
    counts[2] = BA.sum() * nC
    counts[3] = Bi_AB.sum() * nC
    counts[4] = AC.sum() * nB
    counts[5] = AB.T.dot(AC).sum()
    counts[6] = BA.dot(AC).sum()
    counts[7] = Bi_BA.dot(AC).sum()
    counts[8] = CA.sum() * nB
    counts[9] = CA.dot(AB).sum()
    counts[10] = CA.dot(BA.T).sum()
    counts[11] = CA.dot(Bi_AB).sum()
    counts[12] = Bi_AC.sum() * nB
    counts[13] = Bi_CA.dot(AB).sum()
    counts[14] = BA.dot(Bi_AC).sum()
    counts[15] = Bi_BA.dot(Bi_AC).sum()
    counts[16] = BC.sum() * nA
    counts[17] = AB.dot(BC).sum()
    counts[18] = BA.T.dot(BC).sum()
    counts[19] = Bi_AB.dot(BC).sum()
    counts[20] = AC.dot(BC.T).sum()
    counts[21] = AB.dot(BC).multiply(AC).sum()
    counts[22] = BA.dot(AC).multiply(BC).sum()
    counts[23] = Bi_AB.dot(BC).multiply(AC).sum()
    counts[24] = BC.dot(CA).sum()
    counts[25] = tr(AB.dot(BC).dot(CA))
    counts[26] = BC.dot(CA).multiply(BA).sum()
    counts[27] = BC.dot(CA).multiply(Bi_BA).sum()
    counts[28] = BC.dot(Bi_CA).sum()
    counts[29] = AB.dot(BC).multiply(Bi_AC).sum()
    counts[30] = BA.dot(Bi_AC).multiply(BC).sum()
    counts[31] = Bi_BA.dot(Bi_AC).multiply(BC).sum()
    counts[32] = CB.sum() * nA
    counts[33] = AB.dot(CB.T).sum()
    counts[34] = CB.dot(BA).sum()
    counts[35] = CB.dot(Bi_BA).sum()
    counts[36] = AC.dot(CB).sum()
    counts[37] = AC.dot(CB).multiply(AB).sum()
    counts[38] = tr(AC.dot(CB).dot(BA))
    counts[39] = AC.dot(CB).multiply(Bi_AB).sum()
    counts[40] = CA.T.dot(CB).sum()
    counts[41] = CA.dot(AB).multiply(CB).sum()
    counts[42] = CB.dot(BA).multiply(CA).sum()
    counts[43] = CA.T.dot(CB).multiply(Bi_AB).sum()
    counts[44] = Bi_AC.dot(CB).sum()
    counts[45] = Bi_AC.dot(CB).multiply(AB).sum()
    counts[46] = CB.dot(BA).multiply(Bi_CA).sum()
    counts[47] = Bi_CA.dot(Bi_AB).multiply(CB).sum()
    counts[48] = Bi_CB.sum() * nA
    counts[49] = AB.dot(Bi_BC).sum()
    counts[50] = Bi_CB.dot(BA).sum()
    counts[51] = Bi_AB.dot(Bi_BC).sum()
    counts[52] = AC.dot(Bi_CB).sum()
    counts[53] = AB.dot(Bi_BC).multiply(AC).sum()
    counts[54] = BA.dot(AC).multiply(Bi_BC).sum()
    counts[55] = Bi_AB.dot(Bi_BC).multiply(AC).sum()
    counts[56] = Bi_BC.dot(CA).sum()
    counts[57] = CA.dot(AB).multiply(Bi_CB).sum()
    counts[58] = CA.dot(BA.T).multiply(Bi_CB).sum()
    counts[59] = Bi_CB.dot(Bi_BA).multiply(CA).sum()
    counts[60] = Bi_AC.dot(Bi_CB).sum()
    counts[61] = Bi_AC.dot(Bi_CB).multiply(AB).sum()
    counts[62] = Bi_BC.dot(Bi_CA).multiply(BA).sum()
    counts[63] = tr(Bi_AB.dot(Bi_BC).dot(Bi_CA))

    return counts
