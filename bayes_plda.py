#!/home/pengzhiyuan05/.conda/envs/pykaldi/bin/python

# 2021 jerrypeng1937@gmail.com

# This script performs unsupervised bayesian PLDA adaptation following the paper:
#   Unsupervised Bayesian Adaptation of PLDA for Speaker Verification
#
# The critical inputs are:
#       <embed_rspecifier>: per-utt in-domain vector
#       <num_spk>: an approximation of speakers in the data
#       [<pretrained_plda_rxfilename>]: out-of-domain PLDA; if not given, will initialize it as cosine

# The output is:
#       <plda_wxfilename>: where to save adapted PLDA

# attention:
# we found that numpy matrix multiplication requires large amount mem usage, e.g.,
# using np.dot or the @ operator, which may cause core dump
# a current workaround is to replace them by np.einsum, as we did.

# the variable naming follows the paper's notation.


from __future__ import print_function, division
from tqdm import tqdm
import numpy as np
from numpy.linalg import inv
import math
from scipy.special import softmax
from utils import ConfigBase
from dataclasses import dataclass
from pathlib import Path
from base import PldaBase


@dataclass
class Config(ConfigBase):
    prior_within: float = 2.0
    prior_between: float = 2.0
    rand_seed: int = 0
    num_warmup_iter: int = 3
    num_spk: int = -1
    pretrained_plda_rxfilename: Path = Path("")
    embed_rspecifier: str = ""
    plda_wxfilename: Path = Path("")
    verbose: str = 'debug'
    utt2spk_fn: Path = Path("")
    num_iter: int = 9

    def normalize(self):
        super().normalize()
        if self.utt2spk_fn == Path(""):
            assert self.num_spk > 0, f"specify a proper num_spk ({self.num_spk})"
        assert self.num_iter >= 0


class BayesPlda(PldaBase):
    def __init__(self, logger):
        super().__init__(logger)

    def init_stats(self, embed_rspecifier, utt2spk_fn, num_spk):
        # load stats related to embeddings
        self.uttids, self.x_nd = self.read_embed(embed_rspecifier)
        self.num_sample = self.x_nd.shape[0]
        mu = np.mean(self.x_nd, axis=0)
        norm = np.linalg.norm(mu)
        if norm > 1e-2:
            self.warning(f"norm of the mean of input vec: {embed_rspecifier} is {norm:.2f} which is larger than 1e-2")
            self.warning("this is probably bad for plda adaptation but its ok to train")
        self.embed_dim = self.x_nd.shape[1]
        self.Rx_dd = (self.x_nd.T @ self.x_nd) / self.num_sample
        self.Rx_dd = self.force_symmetry(self.Rx_dd)
        # load stats related to speaker assignment z_nm
        self.num_spk = num_spk
        if config.utt2spk_fn == Path(""):
            self.info("random initializing speaker assignments")
            alpha = 1.0 / (1.0 + math.log(1 + self.num_spk)) * np.ones(self.num_spk)
            self.z_nm = np.random.default_rng(seed=config.rand_seed).dirichlet(alpha, size=self.num_sample)
        else:
            self.info(f"initializing speaker assignments from {config.utt2spk_fn}")
            self.z_nm = self.load_z_nm_from_file(config.utt2spk_fn, idx2utt=self.uttids)

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

    def update_stats(self):
        # Order 1
        self.N_m = np.mean(self.z_nm, axis=0)
        self.r_md = (self.z_nm.T @ self.x_nd) / (1.0 + self.num_sample * self.N_m.reshape(-1, 1))
        self.phi_inv_mdd = np.zeros((self.num_spk, self.embed_dim, self.embed_dim))
        self.y_md = np.zeros((self.num_spk, self.embed_dim))
        self.yy_mdd = np.zeros((self.num_spk, self.embed_dim, self.embed_dim))
        for i in range(self.num_spk):
            # Order 2
            self.phi_inv_mdd[i] = self.B + self.num_sample * self.N_m[i] * self.W
            self.phi_inv_mdd[i] = self.force_symmetry(inv(self.phi_inv_mdd[i]))
            # Order 3
            self.y_md[i] = self.phi_inv_mdd[i] @ (self.B @ self.mu + self.W @ self.r_md[i] * (self.num_sample * self.N_m[i]))
            # Order 4
            y_d = self.y_md[i].reshape(-1, 1)
            self.yy_mdd[i] = self.phi_inv_mdd[i] + y_d @ y_d.T
        # Order 5
        self.ry_d = np.mean(self.y_md, axis=0)
        self.Ry_dd = np.sum(self.N_m.reshape(-1, 1, 1) * self.yy_mdd, axis=0)
        self.Ryy_dd = np.mean(self.yy_mdd, axis=0)
        self.Rxy_dd = (self.N_m * self.r_md.T) @ self.y_md

    def update_model(self, prior_within, prior_between):
        # Order 6
        # upadte within-spk covariance
        prior_within_weight = prior_within / (prior_within + 1.0)
        Gw = self.Rx_dd - self.Rxy_dd - self.Rxy_dd.T + self.Ry_dd
        Gw = self.force_symmetry(Gw)
        W_inv = (1.0 - prior_within_weight) * Gw + \
            prior_within_weight * (self.force_symmetry(inv(self.W0)))
        W = inv(W_inv)
        # update global mean
        prior_between_weight = prior_between / (prior_between + 1.0)
        mu = (1.0 - prior_between_weight) * self.ry_d
        # update between-spk covariance
        B_inv = (1.0 - prior_between_weight) * self.Ryy_dd + \
            prior_between_weight * self.force_symmetry(inv(self.B0))
        B_inv -= mu.reshape(-1, 1) @ mu.reshape(-1, 1).T
        B = inv(self.force_symmetry(B_inv))

        self.W = W
        self.mu = mu
        self.B = B

    def update_z_nm(self):
        bias_m = np.trace(self.W @ self.phi_inv_mdd, axis1=1, axis2=2)  # (m)
        C = np.linalg.cholesky(self.W)
        x_nd = np.einsum("ij,jk->ik", self.x_nd, C)
        y_md = np.einsum("ij,jk->ik", self.y_md, C)
        for i in tqdm(range(self.num_sample)):
            diff = x_nd[i] - y_md  # (m, d)
            dist = np.sum(diff ** 2, axis=-1)
            logits = -0.5 * (bias_m + dist)
            self.z_nm[i] = softmax(logits)

    def load_z_nm_from_file(self, utt2spk_fn, idx2utt):
        """ load spk assignments from file
            note that the order of utts in utt2spk_fn and embed_rspecifier
            may be different, as such idx2utt is required for reordering
        """
        assert len(idx2utt) == self.num_sample
        utt2cltidx = {}
        clt2idx = {}
        with open(utt2spk_fn, 'r') as reader:
            for line in reader:
                uttid, cltid = line.rstrip().split()
                if cltid not in clt2idx:
                    clt2idx[cltid] = len(clt2idx)
                utt2cltidx[uttid] = clt2idx[cltid]
        if self.num_spk < len(clt2idx):
            self.info(f"inferring num_spk from the file {utt2spk_fn}")
            self.num_spk = len(clt2idx)
        assert self.num_spk >= len(clt2idx)
        z_nm = np.zeros((self.num_sample, self.num_spk))
        for idx, uttid in enumerate(idx2utt):
            cltid = utt2cltidx[uttid]
            z_nm[idx][cltid] = 1.0
        return z_nm

    def warmup(self, num_iter):
        """ update only speaker assignments """
        for i in range(num_iter):
            self.debug(f"warm-up iter: {i}")
            self.debug("update_stats")
            self.update_stats()
            self.debug("update speaker assignments")
            self.update_z_nm()

    def train(self, config):
        np.random.seed(seed=config.rand_seed)
        self.init_stats(config.embed_rspecifier, config.utt2spk_fn, config.num_spk)
        self.init_model(config.pretrained_plda_rxfilename)
        self.warmup(config.num_warmup_iter)
        for i in range(config.num_iter):
            self.debug(f"iter: {i}")
            self.debug("update_stats")
            self.update_stats()
            self.debug("update model")
            self.update_model(config.prior_within, config.prior_between)
            self.debug("update speaker assignments")
            self.update_z_nm()
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
    plda = BayesPlda(logging)
    plda.train(config)
