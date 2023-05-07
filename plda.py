#!/home/pengzhiyuan05/.conda/envs/pykaldi/bin/python

# 2022 jerrypeng1937@gmail.com

# two-covariance PLDA
# see my note: https://km.sankuai.com/page/1223471074
# for the formuations


from __future__ import print_function, division
import numpy as np
from base import PldaBase
from utils import ConfigBase
from dataclasses import dataclass
from pathlib import Path
from numpy.linalg import inv


@dataclass
class Config(ConfigBase):
    pretrained_plda_rxfilename: Path = Path("")
    embed_rspecifier: str = ""
    plda_wxfilename: Path = Path("")
    verbose: str = 'debug'
    utt2spk_fn: Path = Path("")
    num_iter: int = 9

    def normalize(self):
        super().normalize()
        assert self.num_iter >= 0


class Plda(PldaBase):
    def __init__(self, logger):
        super().__init__(logger)

    def init_stats(self, embed_rspecifier, utt2spk_fn):
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
        # load stats related to speaker assignment z_n
        assert utt2spk_fn != Path(""), "do not support unsupervised training"
        self.z_n = self.load_z_n_from_file(utt2spk_fn, idx2utt=self.uttids)

    def init_model(self, plda_rxfilename):
        """ pls call init_stats first """
        if plda_rxfilename == Path(""):
            self.info("init model from cosine scoring")
            W = np.eye(self.embed_dim)
            B = np.eye(self.embed_dim)
            mu = np.zeros(self.embed_dim)
        else:
            self.info(f"init model from {plda_rxfilename}")
            W, B, mu = self.read(plda_rxfilename)
        self.debug(f"mu.shape: {mu.shape}")
        self.W, self.B, self.mu = W, B, mu
        assert self.embed_dim == len(self.mu)

    def load_z_n_from_file(self, utt2spk_fn, idx2utt):
        """ load spk assignments from file
            note that the order of utts in utt2spk_fn and embed_rspecifier
            may be different, as such idx2utt is required for reordering
        """
        z_n = np.zeros(self.num_sample)
        assert len(idx2utt) == self.num_sample
        utt2cltidx = {}
        clt2idx = {}
        with open(utt2spk_fn, 'r') as reader:
            for line in reader:
                uttid, cltid = line.rstrip().split()
                if cltid not in clt2idx:
                    clt2idx[cltid] = len(clt2idx)
                utt2cltidx[uttid] = clt2idx[cltid]
        self.num_spk = len(clt2idx)
        self.info(f"in total {self.num_spk} speakers for training")
        for idx, utt in enumerate(idx2utt):
            z_n[idx] = utt2cltidx[utt]
        return z_n

    def update_stats(self):
        # Order 1
        # create N_m
        self.N_m = np.zeros(self.num_spk)
        for z in self.z_n:
            self.N_m[int(z)] += 1
        # create r_md
        self.r_md = np.zeros((self.num_spk, self.embed_dim))
        for i, z in enumerate(self.z_n):
            self.r_md[int(z)] += self.x_nd[i]
        self.phi_inv_mdd = np.zeros((self.num_spk, self.embed_dim, self.embed_dim))
        self.y_md = np.zeros((self.num_spk, self.embed_dim))
        self.yy_mdd = np.zeros((self.num_spk, self.embed_dim, self.embed_dim))
        for i in range(self.num_spk):
            # Order 2
            self.phi_inv_mdd[i] = self.B + self.N_m[i] * self.W
            self.phi_inv_mdd[i] = self.force_symmetry(inv(self.phi_inv_mdd[i]))
            # Order 3
            self.y_md[i] = self.phi_inv_mdd[i] @ (self.B @ self.mu + self.W @ self.r_md[i])
            # Order 4
            y_d = self.y_md[i].reshape(-1, 1)
            self.yy_mdd[i] = self.phi_inv_mdd[i] + y_d @ y_d.T
        # Order 5
        self.ry_d = np.mean(self.y_md, axis=0)
        self.Ry_dd = np.sum(self.N_m.reshape(-1, 1, 1) * self.yy_mdd, axis=0) / self.num_sample
        self.Ryy_dd = np.mean(self.yy_mdd, axis=0)
        self.Rxy_dd = (self.r_md.T / self.num_sample) @ self.y_md

    def update_model(self):
        # Order 6
        Gw = self.Rx_dd - self.Rxy_dd - self.Rxy_dd.T + self.Ry_dd
        Gw = self.force_symmetry(Gw)
        W = self.force_symmetry(inv(Gw))
        mu = self.ry_d
        Gb = self.Ryy_dd - mu.reshape(-1, 1) @ mu.reshape(-1, 1).T
        Gb = self.force_symmetry(Gb)
        B = self.force_symmetry(inv(Gb))

        self.W = W
        self.mu = mu
        self.B = B

    def train(self, config):
        self.init_stats(config.embed_rspecifier, config.utt2spk_fn)
        self.init_model(config.pretrained_plda_rxfilename)
        for i in range(config.num_iter):
            self.debug(f"iter: {i}")
            self.debug("updating stats")
            self.update_stats()
            self.debug("updating model")
            self.update_model()
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
    plda = Plda(logging)
    plda.train(config)
