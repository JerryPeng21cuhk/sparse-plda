#!/home/pengzhiyuan05/.conda/envs/pykaldi/bin/python

# 2022 jerrypeng1937@gmail.com

# add sparsity constraint on two-covariance PLDA


from __future__ import print_function, division
import numpy as np
from plda import Plda
from plda import Config as PldaConfig
from dataclasses import dataclass
from numpy.linalg import inv
from pathlib import Path
import math


@dataclass
class Config(PldaConfig):
    prior_within: float = 0.00
    prior_between: float = 0.00
    sparse_penalty: float = 0.01
    sparse_within_covar: bool = False
    sparse_between_covar: bool = True


class SparsePlda(Plda):
    def __init__(self, logger):
        super().__init__(logger)

    def init_model(self, plda_rxfilename):
        """ pls call init_stats first """
        if plda_rxfilename == Path(""):
            self.info("init model from cosine scoring")
            W = np.eye(self.embed_dim)
            B = np.eye(self.embed_dim)
            mu = np.zeros(self.embed_dim)
        else:
            self.info(f"init model from {plda_rxfilename}")
            W, B, mu = self.read_plda(plda_rxfilename)
        self.debug(f"mu.shape: {mu.shape}")
        self.W, self.B, self.mu = W, B, mu
        # priors
        self.W0, self.B0 = np.copy(W), np.copy(B)
        assert self.embed_dim == len(self.mu)

    def update_model(self, config):
        prior_within_weight = config.prior_within / (config.prior_within + 1.0)
        prior_between_weight = config.prior_between / (config.prior_between + 1.0)
        I = np.eye(self.embed_dim)
        # # Order 6
        Gw = self.Rx_dd - self.Rxy_dd - self.Rxy_dd.T + self.Ry_dd
        if config.sparse_within_covar:
            Gw = self.force_symmetry(Gw+I)
            W = self.force_sparse(Gw, self.W, sparse_penalty=config.sparse_penalty)
            W = W * (1.0 - prior_within_weight) + I * prior_within_weight
        else:
            W = self.force_symmetry(inv(Gw+I))
        mu = self.ry_d
        Gb = self.Ryy_dd - mu.reshape(-1, 1) @ mu.reshape(-1, 1).T
        if config.sparse_between_covar:
            Gb = self.force_symmetry(Gb+I)
            B = self.force_sparse(Gb, self.B, sparse_penalty=config.sparse_penalty)
            B = B * (1.0 - prior_between_weight) + I * prior_between_weight
        else:
            B = self.force_symmetry(inv(Gb+I))
        self.debug(f"sparsity rate of W: {self.measure_sparsity(W):.2f}")
        self.debug(f"sparsity rate of B: {self.measure_sparsity(B):.2f}")

        self.W = W
        self.mu = mu
        self.B = B

    def force_sparse(self, G, B, u=0.1, sparse_penalty=0.001, eps=1e-5, max_iter=200):
        # initialization
        B_old = np.copy(B)
        A_old = np.copy(B)
        Phi = np.zeros_like(B)
        # projected gradient descent method to update B_new, B_old
        for i in range(max_iter):
            inner_max_iter = 1000
            max_diff = 100.0
            # inner loop for PSD B
            for j in range(inner_max_iter):
                G_inv = self.force_symmetry(inv(G))
                dL = B_old - G_inv - Phi + u * (B_old - A_old)
                B_new = B_old - dL / (1.0+u)
                evals, U = np.linalg.eigh(B_new)
                evals[evals < 0.0] = 0.0
                B_new = self.force_symmetry(U @ np.diag(evals) @ U.T)
                diff = (abs(B_new - B_old)).max()
                if diff > max_diff:
                    self.debug(f"  sub-iter: {iter1}, update diff: {diff:.4f}")
                B_old = B_new
                if diff < eps:
                    break
            # the rest is for sparse B
            # update A_new
            # dG = Phi + u * (A_old - B_new)
            A_new = B_new - Phi / u
            # compute off-diagonal part
            absA = abs(A_new) - sparse_penalty / u
            absA[absA < 0] = 0
            A_new = np.sign(A_new) * absA
            A_old = A_new
            # update Phi
            Phi = Phi + u * (A_new - B_new)
            # stop criterion
            diff = (abs(A_new - B_new)).max()
            self.debug(f"iter {i}: diff between A and B is {diff}")
            if diff < eps:
                break
        # B_new[B_new < eps] = 0.0
        return B_new

    @staticmethod
    def measure_sparsity(mat, eps=1e-4):
        num_zeros = (abs(mat) < eps).sum()
        m, n = mat.shape
        sp_rate = float(num_zeros) / (m * n)
        return sp_rate

    def train(self, config):
        self.init_stats(config.embed_rspecifier, config.utt2spk_fn)
        self.init_model(config.pretrained_plda_rxfilename)
        for i in range(config.num_iter):
            self.debug(f"iter: {i}")
            self.debug("updating stats")
            self.update_stats()
            self.debug("updating model")
            self.update_model(config)
        self.save(str(config.plda_wxfilename))


if __name__ == "__main__":
    import logging
    config = Config().parse_args()
    verbose_levels = {'info': logging.INFO,
                      'debug': logging.DEBUG,
                      'warning': logging.WARNING}
    assert config.verbose in verbose_levels
    config.plda_wxfilename.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format="%(levelname)s (%(module)s:%(funcName)s():"
                               "%(filename)s:%(lineno)s) %(message)s",
                        level=verbose_levels[config.verbose])
    plda = SparsePlda(logging)
    plda.train(config)
