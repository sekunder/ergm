import numpy as np

from ergm import ERGM


class MCMC:

    model: ERGM

    def __init__(self, model, n):
        """
        Produce a Markov Chain Monte Carlo (MCMC) instance for graphs with `n` nodes, distributed according to
        distribution `ergm`.

        :param ergm:
        :param n:
        """
        self.model = model
        self.n = n

        self.current_adj = np.zeros(n, n)

    def step(self):
        """
        Take a step along the markov chain

        :return:
        """
        pass

    def sample(self):
        """
        Return the current state of the markov chain

        :return:
        """
        return self.current_adj
