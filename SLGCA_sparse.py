import torch
import time
import random
import numpy as np
import sys
import random
import copy
from preprecess import *
from model import *
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.cluster import KMeans
import datetime

class SLGCA_sparse():
    def __init__(self,
                 adata,
                 device=torch.device('cuda'),
                 learning_rate=0.001,
                 weight_decay=0.00,
                 epochs=500,
                 n_top_genes=5000,
                 dim_output=256,
                 random_seed=41,
                 alpha=10,
                 gama=0.5,
                 beta=0.5,
                 n_neighbors=3,
                 ):
        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.n_top_genes = n_top_genes
        self.random_seed = random_seed
        self.alpha = alpha
        self.gama = gama
        self.beta = beta
        self.n_neighbors = n_neighbors

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata, self.n_top_genes)

        if 'adj' not in adata.obsm.keys():
            construct_interaction_KNN(self.adata, self.n_neighbors)

        if 'feat' not in adata.obsm.keys():
            get_feature_sparse(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        print('Building sparse adjacency matrix ...')
        self.adj = preprocess_adj_sparse(self.adj).to(self.device)

    def train(self):
        self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        print('Begin to train sparse ST data...')
        self.model.train()
        copyemb = None
        loss_list = []

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, self.dec, self.dec_a, self.ret, self.ret_a = self.model(self.features,
                                                                                               self.features_a,
                                                                                               self.adj)

            if epoch == 0:
                copyemb = self.emb

            self.loss_feat = F.mse_loss(self.features, self.emb)
            semi_loss = self.model.contrastive_loss(self.dec, self.dec_a)
            clu_loss = self.correlation_reduction_loss(self.cross_correlation(self.ret.t(), self.ret_a.t()))
            loss = self.alpha * self.loss_feat + self.gama * semi_loss + self.beta * clu_loss

            nan_count = torch.isnan(self.emb).sum()
            print(f"Epoch {epoch}, NaN count in embedding: {nan_count.item()}")
            if nan_count.item() > 0:
                print("NaN detected, restoring previous embedding.")
                self.emb = copyemb
                break

            copyemb = self.emb
            loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # plt.figure(figsize=(10, 5))
        # plt.plot(loss_list, label='Training Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss Over Epochs')
        # plt.legend()
        # plt.show()

        print("Training completed for sparse ST data!")

        with torch.no_grad():
            self.adata.obsm['rec'] = self.emb.detach().cpu().numpy()
            return self.adata

    def cross_correlation(self, Z_v1, Z_v2):
        """
        calculate the cross-view correlation matrix S
        Args:
            Z_v1: the first view embedding
            Z_v2: the second view embedding
        Returns: S
        """
        return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())

    def correlation_reduction_loss(self, S):
        """
        the correlation reduction loss L: MSE for S and I (identical matrix)
        Args:
            S: the cross-view correlation matrix S
        Returns: L
        """
        return torch.diagonal(S).add(-1).pow(2).mean() + self.off_diagonal(S).pow(2).mean()

    def off_diagonal(self, x):
        """
        off-diagonal elements of x
        Args:
            x: the input matrix
        Returns: the off-diagonal elements of x
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()