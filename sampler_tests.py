"""Testing the functionality of the ergm package"""

import sys
import time

import numpy as np
import networkx as nx

from ergm import ERGM
from util import log_msg
from scipy.special import binom

log_msg("BEGIN SCRIPT:", __file__)
log_msg("Testing ergms with adjacency matrices")

p0 = 0.1  # edge density for ER graph
n_nodes = 6
n_samples = 10000
seed = 17289  # Not actually used at the moment
np.random.seed(seed)

m = int(binom(n_nodes, 2))

log_msg("Using networkx to sample Erdos-Renyi random graphs with edge probability p = {}".format(p0))
log_msg("Producing {} samples with {} nodes".format(n_samples, n_nodes))
log_msg("Using seed {} for numpy; using 17 * k for nx random graphs".format(seed))

nx_ER_start = time.time()
nx_ER_list = [nx.gnp_random_graph(n_nodes, p0, seed=17 * k) for k in range(n_samples)]
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
# fs = [lambda g: np.sum(g) / 2]  # undirected graph has symmetric adjacency matrix!
# fs = lambda g: np.array([np.sum(g) / 2])
ergm_ER_model = ERGM(lambda g: np.array([np.sum(g) / 2]), [np.log(p0 / (1 - p0))], directed=False)
ergm_ER_samples, ergm_ER_stats = ergm_ER_model.sample_gibbs(n_nodes, n_samples, print_logs=sys.stdout, g0=g0)
ergm_ER_end = time.time()

log_msg("Elapsed time:", ergm_ER_end - ergm_ER_start, "s")

ergm_ER_list = [nx.from_numpy_array(ergm_ER_samples[:, :, i]) for i in range(ergm_ER_samples.shape[2])]

log_msg("Produced", len(ergm_ER_list), "samples")
log_msg("Also produced a vector of statistics (first five elements shown here):", ergm_ER_stats[:5])
log_msg("Mean number of edges:", ergm_ER_stats.mean())

log_msg("Comparing distributions of edge counts:")

theory_edge_distro = np.array([binom(m, k) * (p0 ** k) * ((1 - p0) ** (m - k)) for k in range(m + 2)])
nx_edge_distro, _ = np.histogram([nx.number_of_edges(G) for G in nx_ER_list], bins=range(m + 2))
ergm_edge_distro, _ = np.histogram([nx.number_of_edges(G) for G in ergm_ER_list], bins=range(m + 2))

nx_edge_distro = nx_edge_distro / n_samples
ergm_edge_distro = ergm_edge_distro / n_samples

log_msg("{:>2} {:15} {:15} {:15}".format("m", "nx prob.", "ergm prob.", "theory prob."))
for degree in range(m + 1):
    log_msg(
        f"{degree:2d} {nx_edge_distro[degree]:15.8f} {ergm_edge_distro[degree]:15.8f} {theory_edge_distro[degree]:15.8f}")
    # log_msg("%2d %20.14f %20.14f %20.14f" % (
    # degree, nx_edge_distro[degree], ergm_edge_distro[degree], theory_edge_distro[degree]))

nx_positive_prob = np.where(nx_edge_distro > 0)
ergm_positive_prob = np.where(ergm_edge_distro > 0)

log_msg("KL Divergence between networkx and true distribution:", np.sum(
    nx_edge_distro[nx_positive_prob] * np.log(nx_edge_distro[nx_positive_prob] / theory_edge_distro[nx_positive_prob])))
log_msg("KL Divergence between ergm and true distribution:    ", np.sum(ergm_edge_distro[ergm_positive_prob] * np.log(
    ergm_edge_distro[ergm_positive_prob] / theory_edge_distro[ergm_positive_prob])))

n_large = 100
m_large = n_large * (n_large - 1) // 2

log_msg("Now testing larger graphs, n =", n_large)

log_msg("Networkx fast_gnp_random_graph with p =", p0)
nx_ER_large_start = time.time()
nx_ER_large_list = [nx.fast_gnp_random_graph(n_large, p0, seed=17 * k) for k in range(n_samples)]
nx_ER_large_end = time.time()

log_msg("Elapsed time:", nx_ER_large_end - nx_ER_large_start, "s")
log_msg("Produced", len(nx_ER_large_list), "samples")

log_msg("Now using ergm gibbs sampler, same parameters")

# g0 = np.random.binomial(1, p0, size=(n_large,n_large))
g0 = (np.random.rand(n_large, n_large) < p0).astype(int)
g0[range(n_large), range(n_large)] = 0
g0 = (g0 + g0.T) // 2
log_msg("Initial state:\n", g0)
log_msg("Initial state has", np.sum(g0) // 2, "edges, expected value", m_large * p0)

ergm_ER_large_start = time.time()
# fs = [lambda g: np.sum(g) / 2]  # undirected graph has symmetric adjacency matrix!
# ergm_ER_large_model = ERGM(lambda g: np.array([np.sum(g) / 2]), [np.log(p0 / (1 - p0))], directed=False)
ergm_ER_large_samples, _ = ergm_ER_model.sample_gibbs(n_large, n_samples, print_logs=sys.stdout, burn_in=200, n_steps=200,
                                                      g0=g0)
ergm_ER_large_end = time.time()

log_msg("Elapsed time:", ergm_ER_large_end - ergm_ER_large_start, "s")

ergm_ER_large_list = [nx.from_numpy_array(ergm_ER_large_samples[:, :, i]) for i in
                      range(ergm_ER_large_samples.shape[2])]

log_msg("Produced", len(ergm_ER_large_list), "samples")

nx_ER_large_avg = np.mean([nx.number_of_edges(G) for G in nx_ER_large_list])
ergm_ER_large_avg = np.mean([nx.number_of_edges(G) for G in ergm_ER_large_list])
theory_large_avg = m_large * p0

log_msg("Avg # of edges")
log_msg("{:10}{:10}{:10}".format("nx.gnp", "ergm", "theory"))
log_msg("{:10.2f}{:10.2f}{:10.2f}".format(nx_ER_large_avg, ergm_ER_large_avg, theory_large_avg))

log_msg("Further sampling from ergm with", n_large, "nodes should skip burn-in phase")
n_further = 1000
ergm_further_samples, _ = ergm_ER_model.sample_gibbs(n_large, n_further, print_logs=sys.stdout, burn_in=200, n_steps=200)
avg_edges = ergm_further_samples.sum() / (2 * n_further)
log_msg("Produced", n_further, "samples, with ", avg_edges, "average # edges")

log_msg("END SCRIPT:", __file__)
