"""
adj pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Throughout this package, graphs are represented by their adjacency matrices, which are simply numpy arrays.

Classes:
    ergmpy
"""
import numpy as np
import math


class ergm:
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

    def sample_gibbs(self, n_nodes, n_samples=1, burnin=100, n_steps=500, g0=None):
        """
        Sample from this ensemble, returning a 3d numpy array which is `n_nodes` x `n_nodes` x `n_samples`.

        :param n_nodes: Number of nodes in the graph
        :param n_samples: Number of samples to return
        :param burnin: Number of burn-in steps
        :param n_steps:
        :param g0:
        :return: A numpy array of integers

        This method uses Gibbs sampling.
        TODO write up some details on the internals of this method
        """
        pass

    def biased_loglikelihood(self, samples):
        """
        Compute the biased log likelihood of `samples`, which is a `n_nodes` x `n_nodes` x `n_samples` numpy array,
        representing `n_samples` adjacency matrices. "Biased" means without computing the log of the partition function.

        :param samples: 3d numpy array, shape `n_nodes` x `n_nodes` x `n_samples`
        :return: a float
        """
        return np.mean([self.logweight(samples[:, :, s_idx]) for s_idx in range(samples.shape[2])])
