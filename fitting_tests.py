"""Testing the simplest version of the parameter estimation, using gradient ascent on the loglikelihood (using sample
estimates of the ensemble mean) """

import sys
import time

import networkx as nx
import numpy as np

from ergm import ERGM
from util import log_msg

log_msg("BEGIN SCRIPT:", __file__)
log_msg("Testing parameter estimation using sample mean of ensemble average")

n_nodes = 25
n_observed = 1000
seed = 17289
np.random.seed(seed)

max_edges = n_nodes * (n_nodes - 1) // 2
max_triangles = n_nodes * (n_nodes - 1) * (n_nodes - 2) // 6

theta_true = np.array([1.0, -10000.0])  # be fairly dense, but avoid triangles (cycles all length > 4)

log_msg("Using ergm to produce", n_observed, "observations of low-triangle graphs")
log_msg("theta =", theta_true)


def edge_triangle(g):
    return np.array([g.sum() / (2 * max_edges), np.trace(g @ g @ g) / (6 * max_triangles)])


ET = ERGM(edge_triangle, theta_true)

g0 = (np.random.rand(n_nodes, n_nodes) < (np.exp(theta_true[0]) / (1 + np.exp(theta_true[0])))).astype(int)
g0[range(n_nodes), range(n_nodes)] = 0
g0 = (g0 + g0.T) // 2
# log_msg("Initial state:\n", g0)
initial_edges = g0.sum() // 2
initial_density = initial_edges / max_edges
initial_triangles = np.trace(g0 @ g0 @ g0)
initial_triangle_density = initial_triangles / max_triangles
log_msg("Initial state has", initial_edges, "edges and", initial_triangles, "triangles")

t_start = time.time()
observed_graphs, observed_stats = ET.sample_gibbs(n_nodes, n_observed, print_logs=sys.stdout, g0=g0,
                                                  burn_in=1500, n_steps=1000)
t_end = time.time()
log_msg("Sampled", observed_graphs.shape[2], "graphs in", t_end - t_start, "s")

avg_stats = observed_stats.mean(axis=1)
log_msg("Average edge density:", avg_stats[0])
log_msg("Average tri. density:", avg_stats[1])
log_msg("Erdos Renyi compare: ", (avg_stats[0] ** 3))

edge_ac1 = np.corrcoef(observed_stats[0, :-1], observed_stats[0, 1:])[0, 1]
tri_ac1 = np.corrcoef(observed_stats[1, :-1], observed_stats[0, 1:])[0,1]

log_msg("Correlation between edge density in successive samples:", edge_ac1)
log_msg("Correlation between triangle density in successive samples:", tri_ac1)

theta0 = np.random.randn(2) * 10 + np.ones(2) * 100
log_msg("Now attempting to estimate parameters")
log_msg("theta0 =", theta0)

MLE = ERGM(edge_triangle, theta0)

fit_start = time.time()
MLE.parameter_estimation(observed_graphs, n_estim_samples=100, print_logs=sys.stdout)
fit_end = time.time()

log_msg("Finished in", fit_end - fit_start, "s")
log_msg("theta hat :", MLE.theta)
log_msg("theta true:", ET.theta)
