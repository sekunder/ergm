"""
a pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Classes:
    ergm: uses adjacency matrix representation for graphs
"""
import numpy as np
import math

from util import index_to_edge, log_msg


class ERGM:
    def __init__(self, stats, params=None, directed=False):
        """
        Construct an ergm with specified vector of statistics. In this ensemble,

        $$ P(G) = \frac{1}{Z} exp(\sum_a k_a(G) \theta_a) $$

        where the functions $k_a$ are specified in `stats`, the coefficients $\theta_a$ are specified by `params`.

        :param stats: a list of functions which take numpy matrices as arguments and return numerical values
        :param params: a list of numerical values, or None to use default values of 0 for all coefficients.
        :param directed: Boolean, whether graphs in this ensemble are directed
        """
        self.stats = stats
        if params is None:
            self.params = np.zeros(len(stats))
        else:
            assert len(params) == len(stats)
            self.params = params
        self.directed = directed

        # some extra bits and bobs
        self.expected_stats = np.zeros(len(stats))
        self.expected_other = {}  # for storing expected values of other stats

        # backend for sampling
        self.current_adj = np.zeros(0)  # when sampling, keep the current state of the MCMC
        self.current_logweight = 0.0  # the logweight of the current adjacency matrix
        self.proposed_logweight = 0.0
        # self.last_evaluated = np.zeros(0)  # maybe will just be a hash; some way of caching computations

    def eval_stats(self, adj):
        """
        Compute the statistics of this ergm on the given adjacency matrix.

        :param adj: adjacency matrix
        :return: numpy array, vector of length `len(self.stats)`
        """
        # TODO implement some hashing/caching to avoid repeated computations
        return np.array([f(adj) for f in self.stats])

    def weight(self, adj):
        """
        Compute the weight of adjacency matrix `adj`. This is the unnormalized probability of `adj` under the ergm.
        :param adj: Adjacency matrix of a graph
        :return: A float
        """
        return math.exp(self.logweight(adj))

    def logweight(self, adj):
        """
        Compute the hamiltonian of adj, i.e. the log of the weight of adj (see `weight`).
        :param adj: Adjacency matrix of a graph
        :return: a float
        """
        return np.sum([theta * k(adj) for theta, k in zip(self.params, self.stats)])

    def hamiltonian(self, adj):
        """
        Returns `self.logweight(adj)`
        :param adj: Adjacency matrix
        :return: a float
        """
        return self.logweight(adj)

    def sample_gibbs(self, n_nodes, n_samples=1, burn_in=None, n_steps=None, g0=None, print_logs=None):
        """
        Sample from this ensemble, returning a 3d numpy array which is `n_nodes` x `n_nodes` x `n_samples`.

        :param n_nodes: Number of nodes in the graph
        :param n_samples: Number of samples to return
        :param burn_in: Number of burn-in steps
        :param n_steps: Number of steps between samples
        :param g0: Initial adjacency matrix to use. Default is previous internal state for sampler, if appropriate
        :param print_logs: where to print logs. Default is None, suppressing output
        :return: A numpy array of integers

        This method uses Gibbs sampling.
        """
        # TODO write up some details on the internals of this method in the docstring
        if g0 is None and self.current_adj.shape[0] == n_nodes:
            log_msg("sample_gibbs: previous adjacency matrix found", out=print_logs)
            pass
        elif g0 is None:
            log_msg("sample_gibbs: using empty graph for initial state", out=print_logs)
            self.current_adj = np.zeros((n_nodes, n_nodes))
            self.current_logweight = self.logweight(self.current_adj)
        else:
            log_msg("sample_gibbs: using provided adjacency matrix for initial state", out=print_logs)
            self.current_adj = g0
            self.current_logweight = self.logweight(g0)

        if burn_in is None:
            burn_in = 10 * (n_nodes ** 2) // 2
        if n_steps is None:
            n_steps = 10 * (n_nodes ** 2) // 2

        log_msg("sample_gibbs: %8d burn-in steps" % burn_in, out=print_logs)
        log_msg("sample_gibbs: %8d steps between samples" % n_steps, out=print_logs)

        samples = np.zeros((n_nodes, n_nodes, n_samples), dtype=int)
        total_steps = burn_in + n_steps * n_samples
        urand = np.random.rand(total_steps)
        edge_sequence = index_to_edge(np.random.choice(range(n_nodes * (n_nodes - 1) // (1 + (not self.directed))),
                                                       size=total_steps, replace=True),
                                      n_nodes, self.directed)

        log_msg("sample_gibbs: beginning MCMC process", out=print_logs)
        for step in range(total_steps):
            # assuming the logweight of the current state is already computed, we just need to compute the new values
            # self.current_adj[edge_sequence[0, step], edge_sequence[1, step]] = ~self.current_adj[edge_sequence[0, step], edge_sequence[1, step]]
            self.toggle_current_edge(edge_sequence[0, step], edge_sequence[1, step])
            self.proposed_logweight = self.logweight(self.current_adj)
            p_flip = 1 / (1 + math.exp(self.current_logweight - self.proposed_logweight))
            # p_flip = 1 / (1 + self.weight(self.current_adj))  # wrong!
            if urand[step] < p_flip:
                # keep the flip; save the logweight
                self.current_logweight = self.proposed_logweight
            else:
                # flip the edge back
                # self.current_adj[edge_sequence[0, step], edge_sequence[1, step]] = ~self.current_adj[
                #     edge_sequence[0, step], edge_sequence[1, step]]
                self.toggle_current_edge(edge_sequence[0, step], edge_sequence[1, step])
            if step >= burn_in and (step - burn_in) % n_steps == 0:
                sample_num = (step - burn_in) // n_steps
                if print_logs is not None and sample_num % (n_samples // 10) == 0:
                    log_msg("sample_gibbs: emitting sample %8d / %8d" % (sample_num, n_samples), out=print_logs)
                samples[:, :, sample_num] = self.current_adj[:, :]

        return samples

    def toggle_current_edge(self, u, v):
        """
        Toggle edge (u,v) in the current adjacency matrix underlying the MCMC sampler

        :param u: first vertex
        :param v: second vertex
        :return: None
        """
        # currently assuming a binary adjacency matrix; this may change in the future
        self.current_adj[u, v] = 1 - self.current_adj[u, v]
        if not self.directed:
            self.current_adj[v, u] = 1 - self.current_adj[v, u]

    def biased_loglikelihood(self, samples):
        """
        Compute the biased log likelihood of `samples`, which is a `n_nodes` x `n_nodes` x `n_samples` numpy array,
        representing `n_samples` adjacency matrices. "Biased" means without computing the log of the partition function.

        :param samples: 3d numpy array, shape `n_nodes` x `n_nodes` x `n_samples`
        :return: a float
        """
        return np.mean([self.logweight(samples[:, :, s_idx]) for s_idx in range(samples.shape[2])])

    def sampler_estimate_expected(self, n, fs=None, n_samples=None, **kwargs):
        """
        Estimates the expected value of $f(G)$ for each $f$ in `fs`, for graphs $G$ with `n` nodes. The estimate is
        computed from `n_samples` (drawn with the gibbs sampler)

        :param n: integer, number of nodes
        :param fs: iterable of functions; default is statistics that define the ergm.
        :param n_samples: integer, number of samples to use for estimation. Default is `n ** 2`

        :return: numpy array of estimated expected values of each function
        """
        if fs is None:
            fs = self.stats
        if n_samples is None:
            n_samples = n ** 2

        samples = self.sample_gibbs(n, n_samples, *kwargs)
        means = np.zeros(len(fs))
        for i, f in enumerate(fs):
            means[i] = np.mean([f(samples[:, :, s_idx]) for s_idx in range(n_samples)])

    def erdosrenyi_estimate_expected(self, n, fs=None, n_samples=None, q=None):
        """
        Estimate the expected value of $f(G)$ for each $f$ in `fs`, for graphs $G$ with `n` nodes. The estimate is
        computed by first drawing a large sample of Erdos-Renyi random graphs, computing their statistics,
        then taking the weighted average according to weights under the ergm.

        :param n: integer, number of nodes
        :param fs: iterable, list of functions; default is statistics that define the ergm
        :param n_samples: integer, the number of ER graphs to generate.
        :param q: edge density of ER samples. Default attempts to match ergm's edge density

        :return: numpy array of expected values of each function
        """
        if fs is None:
            fs = self.stats
        if n_samples is None:
            n_samples = n ** 2
        if q is None:
            q = 0.1  # TODO find a way to approximate edge density in ergm

        # er_samples = np.random.binomial(1, q, size=(n, n, n_samples))
        er_samples = (np.random.rand(n, n, n_samples) < q).astype(int)
        er_samples[range(n), range(n), :] = 0  # clear the diagonals

        # below, we avoid computing each f twice, by performing a single call in sample_stats then using that to
        # compute the corresponding weights
        sample_stats = np.array([[f(er_samples[:, :, s_idx]) for s_idx in range(n_samples)] for f in fs])
        sample_weights = np.exp((sample_stats * self.params).sum())  # TODO check sum axis and broadcast-ability
        sample_weights = sample_weights / sample_weights.sum()

        return (sample_weights * sample_stats).sum()
