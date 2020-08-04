"""
A pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Throughout this package, graphs are represented by their adjacency matrices, which are simply numpy arrays.

Classes:
    ergmpy
"""
import numpy as np



class ergm:
    def __init__(self, stats, params=None, directed=False):
        """
        Construct an ergm with specified vector of statistics. In this ensemble,

        $$ P(A) = \frac{1}{Z} exp(\sum_a k_a(A) \theta_a) $$

        where the functions $k_a$ are specified in `stats`, the coefficients $\theta_a$ are specified by `params`.

        :param stats: a list of functions which take numpy matrices as arguments and return numerical values
        :param params: a list of numerical values, or None to use default values of 0 for all coefficients.
        :param directed: Boolean
        """
        self.stats = stats
        if params is None:
            self.params = np.zeros(len(stats))
        else:
            assert len(params) == len(stats)
            self.params = params
        self.directed = directed

    def weight(self, A):
        """
        Compute the weight of adjacency matrix A. This is the unnormalized probability of A under the ergm.
        :param A: Adjacency matrix of a graph
        :return: A float
        """
        pass

    def logweight(self, A):
        """
        Compute the hamiltonian of A, i.e. the log of the weight of A (see `weight`)
        :param A: Adjacency matrix of a graph
        :return: A float
        """
        pass

    def hamiltonian(self, A):
        """
        Returns `self.logweight(A)`
        :param A: Adjacency matrix
        :return: A float
        """
        return self.logweight(A)
