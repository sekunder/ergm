import numpy as np
import networkx as nx
import scipy.sparse as sp
import pandas as pd

import argparse
import logging
from logging import info, debug, warning

from util import index_to_directed_triplet_motif_matrix, last_sorted_index
# from colorful_motifs import (class_representative, overcount_three_color,
#                              iso_list_AAA, iso_list_AAB, iso_list_ABB, iso_list_ABC,
#                              exact_to_over_AAA, exact_to_over_AAB, exact_to_over_ABB, exact_to_over_ABC)
from colorful_motifs import *


parser = argparse.ArgumentParser()
parser.add_argument('--n_nodes', type=int, default=25, help="Number of nodes in test graph. Will perform exhaustive search, over triples, so don't make this too high.")
parser.add_argument('--n_colors', type=int, default=4, help="Number of colors to use (some colors may have no nodes)")
parser.add_argument('--n_loops', type=int, default=10, help="Will generate this many graphs, new coloring each time")
parser.add_argument('--ER_p', type=float, default=0.3, help="Density of test graph")
parser.add_argument('--loglevel', choices=["debug", "info", "warning", "error", "critical"],
                    default="info", help="Level of verbosity")
parser.add_argument('--output_dir', default='tests', help='Where to dump any matrices that fail')
parser.add_argument('--print_adj', action='store_true', help='If passed, print the adjacency matrix at each loop (not recommended)')

args = parser.parse_args()

# configure logging
loglevel = args.loglevel.upper()
logging.basicConfig(format="%(asctime)s  %(module)s> %(levelname)s: %(message)s", datefmt="%Y %m %d %H:%M:%S", level=getattr(logging, loglevel.upper()))

info("Testing colorful motif counting")

n_loops = args.n_loops

n_nodes = args.n_nodes
n_colors = args.n_colors
p_edge = args.ER_p

n_triplets = n_nodes * (n_nodes - 1) * (n_nodes - 2) // 6

info("n_nodes:  %d", n_nodes)
info("n_colors: %d", n_colors)
info("p_edge:   %f", p_edge)

any_trial_failed = False
for trial in range(n_loops):
    info("Trial %d of %d", trial + 1, n_loops)
    parts = np.random.choice(range(n_nodes + n_colors - 1), size=n_colors - 1, replace=False)
    parts.sort()
    full_parts = np.hstack([[-1], parts, [n_nodes + n_colors - 1]])
    between_bars = np.diff(full_parts)
    partition = np.cumsum(np.hstack([[0], between_bars - np.ones_like(between_bars)]))

    color_to_nodes = {k: list(range(partition[k], partition[k + 1])) for k in range(n_colors)}
    n_c = [len(color_to_nodes[k]) for k in range(n_colors)]
    node_to_color = [last_sorted_index(partition, k) for k in range(n_nodes)]
    info("  Nodes of each color: " + str(n_c))

    G = nx.gnp_random_graph(n_nodes, p_edge, directed=True)
    adj = nx.convert_matrix.to_scipy_sparse_matrix(G)
    sub = {}
    for c0 in range(n_colors):
        for c1 in range(n_colors):
            sub[c0, c1] = adj[partition[c0]:partition[c0 + 1], partition[c1]:partition[c1 + 1]]

    # initialize the true counts dictionary with a tuple for every color combo we'll run in to, and a tuple for every possible graph
    info("  Beginning exhaustive search to categorize all triplets of nodes")
    true_counts_ord_col = {}
    for c0_idx in range(n_colors):
        for c1_idx in range(c0_idx, n_colors):
            for c2_idx in range(c1_idx, n_colors):
                true_counts_ord_col[(c0_idx, c1_idx, c2_idx)] = {tuple(index_to_directed_triplet_motif_matrix(k).ravel()): 0 for k in range(64)}

    ticker_count = n_triplets // 100
    triplets_processed = 0
    print("  > Processing triplets [", end="")
    for i in range(n_nodes):
        ci = node_to_color[i]
        for j in range(i + 1, n_nodes):
            cj = node_to_color[j]
            for k in range(j + 1, n_nodes):
                ck = node_to_color[k]
                # unfortunately, selecting submatrices is stupid, so we'll have to do it in the following stupid way
                temp = sp.vstack([adj[i], adj[j], adj[k]])
                submtx = sp.hstack([temp[:, i], temp[:, j], temp[:, k]]).toarray()
                subtup = tuple(submtx.ravel())
                colortup = (ci, cj, ck)
                true_counts_ord_col[colortup][subtup] += 1
                triplets_processed += 1
                if triplets_processed % ticker_count == 0:
                    print(".", end="")
    print("]")
    print("  Triplets processed:", triplets_processed, "/", n_triplets, "(", 100 * triplets_processed / n_triplets, "% )")

    info('  Preparing dataframes to consolidate counts')
    motif_counts_df = pd.DataFrame([[ct, gt, count] for ct in true_counts_ord_col for gt, count in true_counts_ord_col[ct].items()], columns=["Colors", "Motif", "Count"])
    motif_counts_df["representative"] = [class_representative(ct, gt) for ct, gt in zip(motif_counts_df["Colors"], motif_counts_df["Motif"])]
    consolidated_counts_df = motif_counts_df.groupby(["Colors", "representative"]).agg({"Count": "sum"})

    info("  Preparing to count using matrix algebra")
    matrix_over_counts = {}
    matrix_exact_counts = {}
    for c0 in range(n_colors):
        n0 = n_c[c0]
        for c1 in range(c0, n_colors):
            n1 = n_c[c1]
            for c2 in range(c1, n_colors):
                n2 = n_c[c2]
                matrix_over_counts[(c0, c1, c2)], correction = overcount_three_color(c0, c1, c2, n0, n1, n2, sub[c0, c1], sub[c0, c2], sub[c1, c0], sub[c1, c2], sub[c2, c0], sub[c2, c1])
                matrix_exact_counts[(c0, c1, c2)] = correction.dot(matrix_over_counts[(c0, c1, c2)])

    info("  Testing counts")
    any_tests_failed = False
    total_triplets_counted = 0

    max_triplets = {}
    exact_mismatch = {}
    over_mismatch = {}
    for c0 in range(n_colors):
        n0 = n_c[c0]
        max_triplets[(c0, c0, c0)] = n0 * (n0 - 1) * (n0 - 2) // 6
        for c1 in range(c0, n_colors):
            n1 = n_c[c1]
            if c1 != c0:
                max_triplets[(c0, c0, c1)] = n0 * (n0 - 1) * n1 // 2
                max_triplets[(c0, c1, c1)] = n0 * n1 * (n1 - 1) // 2
            for c2 in range(c1, n_colors):
                n2 = n_c[c2]
                if c2 != c1:
                    max_triplets[(c0, c1, c2)] = n0 * n1 * n2
                debug("    Pattern %d %d %d", c0, c1, c2)
                num_counted = consolidated_counts_df.loc[(c0, c1, c2)]["Count"].sum()
                total_triplets_counted += num_counted
                debug("    Number of triplets of this type counted: %d / %d", num_counted, max_triplets[(c0, c1, c2)])
                exhaustive = consolidated_counts_df.loc[(c0, c1, c2)].to_dict()

                if c0 == c1 == c2:
                    iso_list = get_iso_list("AAA")
                    exact_to_over = exact_to_over_AAA
                elif c0 == c1 != c2:
                    iso_list = get_iso_list("AAB")
                    exact_to_over = exact_to_over_AAB
                elif c0 != c1 == c2:
                    iso_list = get_iso_list("ABB")
                    exact_to_over = exact_to_over_ABB
                else:
                    iso_list = get_iso_list("ABC")
                    exact_to_over = exact_to_over_ABC
                exact_counts = np.array([exhaustive["Count"][tup] for tup in iso_list])
                exact_match = np.all(exact_counts == matrix_exact_counts[(c0, c1, c2)])
                any_tests_failed = any_tests_failed and not exact_match
                debug("    Match?", exact_match)
                if not exact_match:
                    exact_mismatch[(c0, c1, c2)] = np.where(exact_counts != matrix_exact_counts[(c0, c1, c2)])
                    over_mismatch[(c0, c1, c2)] = np.where(exact_to_over.dot(exact_counts) != matrix_over_counts[(c0, c1, c2)])
    info("  Number of triplets counted: %d / %d", total_triplets_counted, n_triplets)
    if not any_tests_failed and total_triplets_counted == n_triplets:
        info("  SUCCESS! Counted everything correctly!!!!")
    elif not any_tests_failed:
        info("  Partial success: Somehow got all motif counts right, but didn't tally them all correctly.")
        any_trial_failed = True
    else:
        warning("  Failure :(")
        print("  Offending color scheme:")
        print(partition)
        print("  Offending matrix:")
        print(adj.toarray())
        any_trial_failed = True

print()
if any_trial_failed:
    warning("Some trial failed! Check the logs above.")
else:
    info("BIG SUCCESS! All loops successfully completed!")
