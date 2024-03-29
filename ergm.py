"""
a pure-python implementation of exponential random graph models (ERGMs). Adapted from https://github.com/jcatw/ergm

Classes:
    ergm: uses adjacency matrix representation for graphs
"""
import time

import numpy as np
import math
from scipy import sparse

from util import index_to_edge, log_msg


class ERGM:
    def __init__(self, stats, params, delta_stats=None, directed=False, use_sparse=False):
        """
        Construct an ergm over binary graphs (i.e. edges present or absent) with specified vector of statistics. In
        this ensemble, the probability density function is given by

        $$ P(G | \\theta) = \\frac{1}{Z} exp(\\sum_a k_a(G) \\theta_a) $$

        where the functions $k_a$ are the components of `stats`, the coefficients $\\theta_a$ are specified by `params`.

        :param stats: a function which takes a graph as an argument and returns a vector of statistics
        :param params: an iterable of numerical values.
        :param delta_stats: a function which takes a binary adjacency matrix and a pair of indices as input, and returns
                            the difference in stats (stats with edge) - (stats without edge)
        :param directed: Boolean, whether graphs in this ensemble are directed
        """
        self.stats = stats
        self.delta_stats = delta_stats
        if self.delta_stats is None:
            self.delta_stats = self._naive_delta_stats
        self.theta = np.array(params)
        self.directed = directed
        self.use_sparse = use_sparse

        # some extra bits and bobs
        # self.expected_stats = np.zeros(len(stats))
        # self.expected_other = {}  # for storing expected values of other stats

        # backend for sampling
        self.current_adj = np.zeros(0)  # when sampling, keep the current state of the MCMC
        self.current_stats = np.zeros(0)
        self.current_logweight = 0.0  # the logweight of the current adjacency matrix
        self.proposed_stats = np.zeros(0)
        self.proposed_logweight = 0.0
        # self.last_evaluated = np.zeros(0)  # maybe will just be a hash; some way of caching computations

        # self._Z = dict()

    def _initialize_dense_adj(self, n, reset_stats=False):
        """Initialize self.current_adj to an `n x n` zeros matrix."""
        self.current_adj = np.zeros((n, n), dtype=int)
        if reset_stats:
            self.current_stats = self.stats(self.current_adj)
            # self.current_logweight = np.dot(self.current_stats, self.theta)
            self.current_logweight = self.theta.dot(self.current_stats)

    def _initialize_sparse_adj(self, n, prealloc_coords=None, prealloc_data=None, reset_stats=True):
        """
        Initialize self.current_adj as a sparse matrix with preallocated structure, if specified.
        Specifying a structure will speed up the sampling, since every time an edge is toggled,
        it won't have to reallocate space for the underlying data of the matrix.

        :param n: Number of nodes; generates

        :param prealloc_coords: The matrix coordinates which are to be explicitly stored (even if they are zeros).
            Passed as a `2 x m` numpy array, where `m` is the number of preallocated edges.

        :param prealloc_data: The data to be stored in the matrix (if None, defaults to zeros)

        :param reset_stats: If true, compute the statistics of the new matrix and save the results (default True)

        :return: Does not return anything; modifies the internal state of the object
        """
        if prealloc_coords is None:
            self.current_adj = sparse.csr_matrix((n, n), dtype=int)  # set up an empty matrix
        else:
            m = prealloc_coords.shape[1]
            if prealloc_data is None:
                prealloc_data = np.zeros(m, dtype=int)
            self.current_adj = sparse.csr_matrix((prealloc_data, (prealloc_coords[0], prealloc_coords[1])), shape=(n, n))
        if reset_stats:
            self.current_stats = self.stats(self.current_adj)
            self.current_logweight = self.theta.dot(self.current_stats)

    def _sparse_adj_drop_zeros(self):
        """
        Compress the current adjacency matrix by dropping the explicit 0s stored in the data.
        """
        # turns out there's an in-place function for this!
        self.current_adj.eliminate_zeros()

    def _sparse_adj_add_zeros(self, alloc_coords, alloc_data=None):
        """
        Allocate space for the specified data in the specified locations, storing explicit zeros if need be.
        :param alloc_coords: A `2 x m` numpy array of matrix indices to allocate
        :param alloc_data: Data
        :return:
        """
        cur_shape = self.current_adj.shape()
        I, J, V = sparse.find(self.current_adj)
        # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        if alloc_data is None:
            alloc_data = np.zeros(alloc_coords.shape[1])
        self.current_adj = sparse.csr_matrix((np.stack(V, alloc_data),
                                              (np.stack(I, alloc_coords[0]), np.stack(J, alloc_coords[1]))),
                                             shape=cur_shape)

    def eval_stats(self, adj):
        """
        Compute the statistics of this ergm on the given adjacency matrix.

        :param adj: adjacency matrix
        :return: numpy array, vector of length `len(self.stats)`
        """
        # TODO implement some hashing/caching to avoid repeated computations
        # return np.array([f(adj) for f in self.stats])
        return self.stats(adj)

    def _naive_delta_stats(self, g, u, v, recompute_current=False):
        """
        Compute the difference between the stats of the current adjacency matrix and the adjacency matrix with edge
        `(u,v)` toggled (returns $k(g) - k(g')$).

        :param g: graph in question
        :param u: first vertex
        :param v: second vertex
        :param recompute_current: If true, compute the stats of the current adjacency matrix
        :return: the difference $k(g) - k(g')$

        This is fairly inefficient as it will compute the stats of the entire matrix, possibly twice.
        """
        # TODO think through a way to keep track of whether current_stats is actually current
        if recompute_current:
            self.current_stats = self.eval_stats(g)
        self._toggle_current_edge(u, v)  # flip the edge
        new_stats = self.eval_stats(g)   # compute the new stats
        self._toggle_current_edge(u, v)  # flip the edge back
        return self.current_stats - new_stats

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
        # return np.sum([theta * k(adj) for theta, k in zip(self.theta, self.stats)])
        # return np.dot(self.stats(adj), self.theta)
        return self.theta.dot(self.stats(adj))

    def hamiltonian(self, adj):
        """
        Returns `self.logweight(adj)`
        :param adj: Adjacency matrix
        :return: a float
        """
        return self.logweight(adj)

    def sample_gibbs(self, n_nodes, n_samples=1, burn_in=None, n_steps=None, g0=None, print_logs=None):
        """
        Sample from this ensemble, returning a 3d numpy array which is `n_nodes x n_nodes x n_samples` and a 2d
        numpy array which is `d x n_samples`, where `d` is the number of statistics. The second array stores the
        statistics of each sample, to avoid recomputing them (e.g. in parameter estimation)

        :param n_nodes: Number of nodes in the graph
        :param n_samples: Number of samples to return
        :param burn_in: Number of burn-in steps
        :param n_steps: Number of steps between samples
        :param g0: Initial adjacency matrix to use. Default is previous internal state for sampler, if appropriate
        :param print_logs: where to print logs. Default is None, suppressing output
        :return: A numpy array of integers (the adjacency matrices) and a numpy array of floats (the statistics)
                 and a dictionary of diagnostic info 

        This method uses Gibbs sampling.
        """
        # TODO write up some details on the internals of this method in the docstring
        diagnostics = {}

        # Compute number of steps this MC will take
        if burn_in is None:
            if self.current_adj.shape[0] == n_nodes:
                burn_in = 0
            else:
                burn_in = 3 * int(math.ceil(n_nodes * math.log(n_nodes))) * len(self.theta)
            # above is based on some rough estimates/simulations
        if n_steps is None:
            n_steps = int(math.ceil(math.log(n_nodes))) * len(self.theta) * 2
        total_steps = burn_in + n_steps * n_samples  # total steps taken by markov chain

        # initialize output arrays
        if self.use_sparse:
            samples = np.array([sparse.csr_matrix((n_nodes, n_nodes), dtype=int) for _ in range(n_samples)])
        else:
            samples = np.zeros((n_nodes, n_nodes, n_samples), dtype=int)
        sample_stats = np.zeros((self.theta.shape[0], n_samples))

        # determine the sequence of edges to visit
        urand = np.random.rand(total_steps)
        pflip = np.zeros_like(urand)  # the probability of flipping the edge at each step (aka proposal probability)
        accept = np.zeros_like(urand, dtype=bool)  # whether the edge was flipped at each step
        accept_rate = np.zeros(n_samples)  # what fraction of steps were actual moves from one sample to the next
        # rand_indexes = np.random.choice(range(n_nodes * (n_nodes - 1) // (1 + (not self.directed))), size=total_steps, replace=True)
        rand_indexes = np.random.randint(0, n_nodes * (n_nodes - 1) // (1 + (not self.directed)), size=total_steps)
        edge_sequence = index_to_edge(rand_indexes, n_nodes, self.directed)
        # TODO There might be a way to speed up indexing into a CSR matrix by reformatting the edge sequence

        if g0 is None and self.current_adj.shape[0] == n_nodes:
            log_msg("sample_gibbs: previous adjacency matrix found", out=print_logs)
            # burn_in = 0  # we're picking up where we left off
            # pass
        elif g0 is None:
            log_msg("sample_gibbs: using empty graph for initial state", out=print_logs)
            # self.current_adj = np.zeros((n_nodes, n_nodes))
            # self.current_stats = self.stats(self.current_adj)
            # self.current_logweight = np.dot(self.current_stats, self.theta)
            # TODO enable random initialization
            if self.use_sparse:
                self._initialize_sparse_adj(n_nodes, edge_sequence, reset_stats=True)
            else:
                self._initialize_dense_adj(n_nodes, reset_stats=True)
            self.proposed_stats = np.zeros_like(self.current_stats)
            # self.current_logweight = self.logweight(self.current_adj)
        else:
            log_msg("sample_gibbs: using provided adjacency matrix for initial state", out=print_logs)
            self.current_adj = g0
            self.current_stats = self.stats(self.current_adj)
            self.current_logweight = self.theta.dot(self.current_stats)
            self.proposed_stats = np.zeros_like(self.current_stats)
            # self.current_logweight = self.logweight(g0)

        log_msg("sample_gibbs: %8d nodes" % n_nodes, out=print_logs)
        log_msg("sample_gibbs: %8d burn-in steps" % burn_in, out=print_logs)
        log_msg("sample_gibbs: %8d steps between samples" % n_steps, out=print_logs)
        log_msg("sample_gibbs: %8d total steps" % total_steps, out=print_logs)

        log_msg("sample_gibbs: beginning MCMC process", out=print_logs)
        t_start = time.time()
        for step in range(total_steps):
            # assuming the logweight of the current state is already computed, we just need to compute the new values
            delta_k = self.delta_stats(self.current_adj, edge_sequence[0, step], edge_sequence[1, step])
            self.proposed_stats[:] = self.current_stats[:] - delta_k[:]  # the [:] are there to avoid new allocations(?)
            self.proposed_logweight = self.theta.dot(self.proposed_stats)
#             p_flip = 1 / (1 + math.exp(self.theta.dot(delta_k)))
            delta_E = self.theta.dot(delta_k)
            try:
                pflip[step] = 1 / (1 + math.exp(self.theta.dot(delta_k)))
            except OverflowError:
                if delta_E > 0:
                    pflip[step] = 0.0
                else:
                    pflip[step] = 1.0
#             if urand[step] < p_flip:
            if urand[step] < pflip[step]:
                # flip the edge, save the logweight and stats
                self._toggle_current_edge(edge_sequence[0, step], edge_sequence[1, step])
                self.current_stats[:] = self.proposed_stats[:]
                self.current_logweight = self.proposed_logweight
                # avoid modifying self.current_adj, which may be sparse, until we're sure we're flipping the edge.
                accept[step] = True
            if step >= burn_in and (step - burn_in) % n_steps == 0:
                # emit sample
                sample_num = (step - burn_in) // n_steps
                if print_logs is not None and (n_samples < 10 or (sample_num + 1) % (n_samples // 10) == 0):
                    log_msg("sample_gibbs: emitting sample %8d / %8d" % (sample_num + 1, n_samples), out=print_logs)
                # samples[:, :, sample_num] = self.current_adj[:, :]
                if self.use_sparse:
                    # samples[sample_num][:, :] = self.current_adj[:, :]
                    samples[sample_num] = self.current_adj.copy()
                else:
                    samples[:, :, sample_num] = self.current_adj[:, :]
                sample_stats[:, sample_num] = self.current_stats[:]
                accept_rate[sample_num] = accept[(step - n_steps):step].mean()
        t_end = time.time()
        log_msg("sample_gibbs: Sampling finished in", t_end - t_start, "s", out=print_logs)
        diagnostics["proposal"] = pflip
        diagnostics["accept"] = accept
        diagnostics["accept_rate"] = accept_rate

        return samples, sample_stats, diagnostics

    def _toggle_current_edge(self, u, v):
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

    def sampler_estimate_expected(self, n, f_vec=None, n_samples=None, **kwargs):
        """
        Estimates the expected value of $f(G)$, where $f$ is a vector-valued function of graphs, for graphs $G$ with
        `n` nodes. The estimate is computed from `n_samples` (drawn with the gibbs sampler)

        :param n: integer, number of nodes
        :param f_vec: function which takes a graph and returns a vector. default is statistics that define the ergm.
        :param n_samples: integer, number of samples to use for estimation. Default is `n ** 2`

        :return: numpy array of estimated expected values of each function
        """
        if f_vec is None:
            f_vec = self.stats
        if n_samples is None:
            n_samples = n ** 2

        samples, _ = self.sample_gibbs(n, n_samples, **kwargs)
        return np.array([f_vec(samples[:, :, i] for i in range(n_samples))]).mean(axis=0)

    def importance_estimate_expected(self, n, f_vec=None, n_samples=None, q=None):
        """
        Estimate the expected value of $f(G)$ for each $f$ in `f_vec`, for graphs $G$ with `n` nodes. The estimate is
        computed by first drawing a large sample of Erdos-Renyi random graphs, computing their statistics,
        then taking the weighted average according to weights under the ergm.

        :param n: integer, number of nodes
        :param f_vec: iterable, list of functions; default is statistics that define the ergm
        :param n_samples: integer, the number of ER graphs to generate.
        :param q: edge density of ER samples. Default attempts to match ergm's edge density

        :return: numpy array of expected values of each function
        """
        # TODO implement this correctly, currently it is (at least) missing a correction factor for the ER distribution
        if f_vec is None:
            f_vec = self.stats
        if n_samples is None:
            n_samples = n ** 2
        if q is None:
            q = 0.1  # TODO find a way to approximate edge density in ergm

        # er_samples = np.random.binomial(1, q, size=(n, n, n_samples))
        er_samples = (np.random.rand(n, n, n_samples) < q).astype(int)
        er_samples[range(n), range(n), :] = 0  # clear the diagonals

        # below, we avoid computing each f twice, by performing a single call in sample_stats then using that to
        # compute the corresponding weights
        # sample_stats = np.array([[f(er_samples[:, :, s_idx]) for s_idx in range(n_samples)] for f in f_vec])
        sample_stats = np.array([f_vec(er_samples[:, :, s_idx] for s_idx in range(n_samples))])
        sample_weights = np.exp((sample_stats * self.theta).sum())  # TODO check sum axis and broadcast-ability
        sample_weights = sample_weights / sample_weights.sum()

        return (sample_weights * sample_stats).sum()

    def parameter_estimation(self, observed, **kwargs):
        """
        Estimate parameters for the given data. Currently only supports one method, maximum likelihood estimation via sampler.

        :param observed: Data to fit to. Numpy array, either n_nodes x n_nodes or n_nodes x n_nodes x n_samples
        :param kwargs:
        """
        # currently just a wrapper for _MLE_sampler
        return self._MLE_sampler(observed, **kwargs)

    def _MLE_sampler(self, observed, n_estim_samples=100,
                     alpha=1, alpha_rate=0.999, max_iter=1000, x_tol=1e-8,
                     print_logs=None, sampler_logs=None, **kwargs):
        """
        Compute the maximum likelihood estimate (MLE) of parameters for the observed data using gradient ascent on
        the likelihood. The expected value of the current ERGM is estimated by sampling.

        :param observed: Observed graphs, shape n_nodes x n_nodes x n_samples
        :param n_estim_samples: samples to use at each estimation step
        :param alpha: Scale factor of gradient.
        :param max_iter: Maximum number of iterations to take
        :param x_tol: Quit trying to optimize if gradient norm falls below x_tol
        :param print_logs: File pointer, where to print logs (default None to suppress output)
        :param sampler_logs: File pointer, where sampler method should print logs (default None to suppress output)
        :param kwargs: Additional keyword arguments are passed to sampler
        :return: Dictionary including trajectory and such.
        """
        if len(observed.shape) == 2:
            k_obs = self.stats(observed)
            log_msg("MLE_sampler: Passed single graph; observed stats:\n", k_obs, out=print_logs)
        else:
            try:
                all_obs_k = np.array([self.stats(observed[:, :, i]) for i in range(observed.shape[2])])
            except:
                # if sparse arrays were used, observed will be an array of sparse matrices
                all_obs_k = np.array([self.stats(observed[i].tocsr()) for i in range(observed.shape[0])])
            k_obs = all_obs_k.mean(axis=0)
            log_msg("MLE_sampler: Computed stats, resulting shape:", all_obs_k.shape, out=print_logs)
            log_msg("MLE_sampler: average stats:", k_obs, out=print_logs)

        log_msg("MLE_sampler: %8d estimate samples" % n_estim_samples, out=print_logs)
        log_msg("MLE_sampler: %8f alpha" % alpha, out=print_logs)
        log_msg("MLE_sampler: %8d max iterations" % max_iter, out=print_logs)
        log_msg("MLE_sampler: %8e L tolerance" % x_tol, out=print_logs)

        # trajectory = np.zeros((max_iter, self.theta.shape[0]))

        n_nodes = observed.shape[0]

        # trajectory of thetas and such
        theta_t = np.zeros((max_iter + 1, *self.theta.shape))
        k_bar_t = np.zeros_like(theta_t)
        grad_t = np.zeros_like(theta_t)
        covar_t = np.zeros((max_iter + 1, *self.theta.shape, *self.theta.shape))

        # local variables of the algorithm
        delta_theta = np.zeros_like(self.theta)
        theta_t[0, :] = self.theta  # store initial value of theta

        # the sample values at each iteration

        # The actual sample graphs are not needed, so for now, I'll just ignore them. If they turn out to be useful,
        # we can sort out how to handle sparse vs. non-sparse ERGMs.
        # if self.use_sparse:
        #     G_samp = np.array([sparse.lil_matrix(0) for _ in range(n_estim_samples)])
        # else:
        #     G_samp = np.zeros((n_nodes, n_nodes, n_estim_samples), dtype=int)
        k_samp = np.zeros((*self.theta.shape, n_estim_samples))

        log_msg(f"{'iter':4} {'|theta(t)|':20} {'|gradient|':20} {'alpha':20} {'|Delta theta|':20}", out=print_logs)
        stopping_criteria = []
        stop_iter = -1
        for step in range(max_iter):
            try:
                _, k_samp[:, :] = self.sample_gibbs(n_nodes, n_estim_samples, print_logs=sampler_logs,
                                                    **kwargs)
                k_bar_t[step, :] = k_samp.mean(axis=1)
                covar_t[step, :, :] = np.cov(k_samp)
                grad_t[step, :] = k_obs - k_bar_t[step, :]
                grad_norm = np.linalg.norm(grad_t[step, :])
                delta_theta[:] = alpha * grad_t[step, :]

                log_msg(
                    f"{step:4d} {np.linalg.norm(theta_t[step, :]):20.8f} {grad_norm:20.8f} {alpha:20.8f} {alpha * grad_norm:20.8f}",
                    out=print_logs)

                theta_t[step + 1, :] = theta_t[step, :] + delta_theta
                self._set_theta(theta_t[step + 1, :], True)

                # update alpha
                alpha = alpha * alpha_rate
                # check stopping criteria
                # TODO implement 2nd order stopping criteria
                if step + 1 == max_iter:
                    stopping_criteria.append("max_iter reached")
                if grad_norm < x_tol:
                    stopping_criteria.append("grad_norm < x_tol")
                if len(stopping_criteria) > 0:
                    stop_iter = step
                    # break
            except KeyboardInterrupt:
                stopping_criteria.append("keyboard interrupt")
                stop_iter = step - 1
                # break
            except Exception as e:
                stopping_criteria.append("unhandled exception: {}".format(e))
                stop_iter = step - 1
                # break
            finally:
                if len(stopping_criteria) > 0:
                    if stop_iter < 0:
                        stop_iter = step
                    break

        trajectory = {"theta": theta_t[:stop_iter, :],
                      "expected stats": k_bar_t[:stop_iter, :],
                      "covariances": covar_t[:stop_iter, :, :],
                      "gradient": grad_t[:stop_iter, :],
                      "stop": stopping_criteria,
                      "stop_iter": stop_iter}
        return trajectory

    def _set_theta(self, theta_new, compute_new=False):
        """
        Change the parameters theta of the ERGM. If compute_new is True, evaluates the statistics of the current
        state of the MC and updates its logweight.

        :param theta_new: new value for parameters
        :param compute_new: if true, compute logweight of the current MC state
        """
        self.theta = theta_new
        if compute_new:
            self.current_logweight = self.logweight(self.current_adj)
