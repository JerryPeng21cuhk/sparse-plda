#!/home/pengzhiyuan05/.conda/envs/pykaldi/bin/python

# 2022 jerrypeng1937@gmail.com


# base class for plda read/write
# using kaldi format

# Note that this script requires the use of pykaldi package
#  for the installation details, see https://github.com/pykaldi/pykaldi


from __future__ import print_function, division
import numpy as np
from numpy.linalg import inv
from kaldi.util.io import xopen
from kaldi.util.table import SequentialVectorReader
from kaldi.base.io import expect_token, write_token
from kaldi.matrix import DoubleMatrix, DoubleVector


class PldaBase(object):
    def __init__(self, logger):
        super().__init__()
        # to print information
        self.warning = logger.warning
        self.info = logger.info
        self.debug = logger.debug

    @staticmethod
    def read_model(plda_rxfilename):
        with xopen(str(plda_rxfilename)) as ki:
            expect_token(ki.stream(), ki.binary, "<Plda>")
            mean = DoubleVector().read_(ki.stream(), ki.binary)
            transform = DoubleMatrix().read_(ki.stream(), ki.binary)
            psi = DoubleVector().read_(ki.stream(), ki.binary)
            expect_token(ki.stream(), ki.binary, "</Plda>")
        mu = mean.numpy()
        p = transform.numpy()
        psi = psi.numpy()
        p = inv(p)
        W = p @ p.T
        B = np.einsum('ij,j,jt->it', p, psi, p.T)
        B = PldaBase.force_symmetry(B)
        # convert covar into precision mat
        W = PldaBase.force_symmetry(inv(W))
        B = PldaBase.force_symmetry(inv(B))
        return W, B, mu

    @staticmethod
    def is_symmetry(mat):
        return (mat == mat.T).all()

    @staticmethod
    def force_symmetry(mat):
        return 0.5 * (mat + mat.T)

    @staticmethod
    def read_embed(embed_rspecifier):
        embeds = []
        uttids = []
        with SequentialVectorReader(embed_rspecifier) as reader:
            for uttid, embed in reader:
                embeds.append(embed.numpy())
                uttids.append(uttid)
        embeds = np.array(embeds, dtype='float64')
        return uttids, embeds

    def save(self, plda_wxfilename):
        # precision to covariance
        B = inv(self.B)
        W = inv(self.W)
        C = np.linalg.cholesky(W)
        P = inv(C)
        B_proj = self.force_symmetry(P @ B @ P.T)
        eig, U = np.linalg.eig(B_proj)
        assert min(eig) >= 0.0
        num_floors = sum(eig == 0.0)
        if num_floors > 0:
            self.warning(f"Floored {num_floors} eigenvalues of between-class variance to zero.")
        transform = U.T @ P
        self.info(f"Top-10 largest eigenvalue of between-class variance in normalized space is: {sorted(eig, reverse=True)[:10]} ")
        # save to file
        with xopen(str(plda_wxfilename), "w") as ko:
            binary = True
            write_token(ko.stream(), binary, "<Plda>")
            DoubleVector(self.mu).write(ko.stream(), binary)
            DoubleMatrix(transform).write(ko.stream(), binary)
            DoubleVector(eig).write(ko.stream(), binary)
            write_token(ko.stream(), binary, "</Plda>")
        return
