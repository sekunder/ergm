"""
adj pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Throughout this package, graphs are represented by their adjacency matrices, which are simply numpy arrays.

Classes:
    ergmpy
"""
import numpy as np
import math

from util import index_to_edge


class ERGM:
    def __init__(self, stats, params=None, directed=False):
        """
        Construct an ergm with specified vector of statistics. In this ensemble,

        $$ P(adj) = \frac{1}{Z} exp(\sum_a k_a(adj) \theta_a) $$

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

    def weight(self, adj):
        """
        Compute the weight of adjacency matrix `adj`. This is the unnormalized probability of `adj` under the ergm.
        :param adj: Adjacency matrix of a graph
        :return: A float
        """
        return math.exp(self.logweight((adj)))

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

    def sample_gibbs(self, n_nodes, n_samples=1, burn_in=None, n_steps=None, g0=None):
        """
        Sample from this ensemble, returning a 3d numpy array which is `n_nodes` x `n_nodes` x `n_samples`.

        :param n_nodes: Number of nodes in the graph
        :param n_samples: Number of samples to return
        :param burn_in: Number of burn-in steps
        :param n_steps: Number of steps between samples
        :param g0: Initial adjacency matrix to use. If `sample_gibbs` has been called before with the same `n_nodes`, this defaults to the
        :return: A numpy array of integers

        This method uses Gibbs sampling.
        TODO write up some details on the internals of this method
        """
        if g0 is None:
            self.current_adj = np.zeros((n_nodes, n_nodes))
        else:
            self.current_adj = g0

        if burn_in is None:
            burn_in = 10 * n_nodes
        if n_steps is None:
            n_steps = 10 * n_nodes

        samples = np.zeros((n_nodes, n_nodes, n_samples), dtype=int)
        total_steps = burn_in + n_steps * n_samples
        urand = np.random.rand(total_steps)
        # idx_sequence = np.random.choice(range(n_nodes * (n_nodes - 1) // (1 + (not self.directed))),
        #                                 size=total_steps)
        edge_sequence = index_to_edge(np.random.choice(range(n_nodes * (n_nodes - 1) // (1 + (not self.directed))),
                                                       size=total_steps),
                                      n_nodes, self.directed)

        for step in range(total_steps):
            p_flip = 1 / (1 + self.weight(self.current_adj))  # TODO double check this
            if urand[step] < p_flip:
                self.current_adj[edge_sequence[0, step], edge_sequence[1, step]] = ~self.current_adj[
                    edge_sequence[0, step], edge_sequence[1, step]]
            if step >= burn_in and (step - burn_in) % n_steps == 0:
                samples[:, :, (step - burn_in) // n_steps] = self.current_adj[:, :]

        return samples

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

        samples = self.sample_gibbs(n, n_samples, kwargs)  # TODO look up how to use kwargs to streamline all this
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

        er_samples = np.random.binomial(2, q, size=(n, n, n_samples))
        er_samples[range(n), range(n), :] = 0  # clear the diagonals

        # below, we avoid computing each f twice, by performing a single call in sample_stats then using that to
        # compute the corresponding weights
        sample_stats = np.array([[f(er_samples[:, :, s_idx]) for s_idx in range(n_samples)] for f in fs])
        sample_weights = np.exp((sample_stats * self.params).sum())  # TODO check sum axis and broadcast-ability
        sample_weights = sample_weights / sample_weights.sum()

        return (sample_weights * sample_stats).sum()
