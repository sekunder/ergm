"""Testing the functionality of the ergm package"""

import datetime
import sys
import time

import networkx as nx
import numpy as np

from ergm import ERGM
from util import log_msg
from scipy.special import binom

log_msg("Testing ergms with adjacency matrices")

p0 = 0.2  # edge density for ER graph
n_nodes = 6
n_samples = 20000
seed = 17289  # Not actually used at the moment
np.random.seed(seed)

m = int(binom(n_nodes, 2))

log_msg("Using networkx to sample Erdos-Renyi random graphs with edge probability p = {}".format(p0))
log_msg("Producing {} samples with {} nodes".format(n_samples, n_nodes))
log_msg("Using seed {}".format(seed))

nx_ER_start = time.time()
nx_ER_list = [nx.gnp_random_graph(n_nodes, p0) for k in range(n_samples)]
nx_ER_end = time.time()

log_msg("Elapsed time:", nx_ER_end - nx_ER_start, "s")
log_msg("Produced", len(nx_ER_list), "samples")

log_msg("Now using ergm gibbs sampler, same parameters")

# g0 = np.random.binomial(1, p0, size=(n_nodes,n_nodes))
g0 = (np.random.rand(n_nodes, n_nodes) < p0).astype(int)
g0[range(n_nodes), range(n_nodes)] = 0
g0 = (g0 + g0.T) // 2
log_msg("Initial state:\n", g0)
log_msg("Initial state has", np.sum(g0) // 2, "edges, expected value", m * p0)

ergm_ER_start = time.time()
fs = [lambda g : np.sum(g) / 2]  # undirected graph has symmetric adjacency matrix!
ergm_ER_model = ERGM(fs, [np.log(p0 / (1 - p0))], directed=False)
ergm_ER_samples = ergm_ER_model.sample_gibbs(n_nodes, n_samples, print_logs=sys.stdout)
ergm_ER_end = time.time()

log_msg("Elapsed time:", ergm_ER_end - ergm_ER_start, "s")

ergm_ER_list = [nx.from_numpy_array(ergm_ER_samples[:, :, i]) for i in range(ergm_ER_samples.shape[2])]

log_msg("Produced", len(ergm_ER_list), "samples")

log_msg("Comparing distributions of edge counts:")


theory_edge_distro = np.array([binom(m, k) * (p0 ** k) * ((1 - p0) ** (m - k)) for k in range(m + 2)])
nx_edge_distro, _ = np.histogram([nx.number_of_edges(G) for G in nx_ER_list], bins=range(m + 2))
ergm_edge_distro, _ = np.histogram([nx.number_of_edges(G) for G in ergm_ER_list], bins=range(m + 2))

nx_edge_distro = nx_edge_distro / n_samples
ergm_edge_distro = ergm_edge_distro / n_samples

log_msg("{:>2} {:20} {:20} {:20}".format("m", "nx prob.", "ergm prob.", "theory prob."))
for degree in range(m + 1):
    # print(f"{degree:2d} {nx_edge_distro[degree]:20.14f} {ergm_edge_distro[degree]:20.14f} {theory_edge_distro[degree]:20.14f}")
    log_msg("%2d %20.14f %20.14f %20.14f" % (degree, nx_edge_distro[degree], ergm_edge_distro[degree], theory_edge_distro[degree]))

nx_positive_prob = np.where(nx_edge_distro > 0)
ergm_positive_prob = np.where(ergm_edge_distro > 0)

log_msg("KL Divergence between networkx and true distribution:", np.sum(nx_edge_distro[nx_positive_prob] * np.log(nx_edge_distro[nx_positive_prob] / theory_edge_distro[nx_positive_prob])))
log_msg("KL Divergence between ergm and true distribution:    ", np.sum(ergm_edge_distro[ergm_positive_prob] * np.log(ergm_edge_distro[ergm_positive_prob] / theory_edge_distro[ergm_positive_prob])))
