import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from dask.dataframe.utils import assert_dask_dtypes
from numba.tests.test_array_exprs import distance_matrix
from torch.backends import cudnn
# from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ot
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist,euclidean,cosine
from scipy.special import softmax
from anndata import AnnData
from scipy.linalg import block_diag
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from tqdm import tqdm


def filter_with_overlap_gene(adata, adata_sc):
    # remove all-zero-valued genes
    # sc.pp.filter_genes(adata, min_cells=1)
    # sc.pp.filter_genes(adata_sc, min_cells=1)

    if 'highly_variable' not in adata.var.keys():
        raise ValueError("'highly_variable' are not existed in adata!")
    else:
        adata = adata[:, adata.var['highly_variable']]

    if 'highly_variable' not in adata_sc.var.keys():
        raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:
        adata_sc = adata_sc[:, adata_sc.var['highly_variable']]

        # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes

    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]

    return adata, adata_sc

def permutation(feature):
    # fix_seed(FLAGS.random_seed)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated


def preprocess(adata,n_top_genes):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def get_feature(adata):

    adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]


    adata.obsm['feat'] = feat

def get_feature_sparse(adata):
    adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

    feat_a = permutation(feat)

    adata.obsm['feat'] = feat

    adata.obsm['feat_a'] = feat_a

def load_data(path = None,sample_list = None , n_top_genes = 5000,n_neighbors=3,label = False):
    adata_list = []
    for i in sample_list:
        print('load: ' + i)
        load_path = os.path.join(path, i)
        adata = sc.read_visium(load_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        # adata = sc.read_h5ad(load_path + '/filtered_feature_bc_matrix.h5ad')
        adata.var_names_make_unique()
        # adata = sc.read_h5ad(load_path + 'filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if label:
            df_meta = pd.read_csv(load_path + "/metadata.tsv", sep="\t")
            de_meta_layer = df_meta['layer_guess']
            adata.obs['ground_truth'] = de_meta_layer.values
            print(i + ' load label done')
            adata = adata[~pd.isnull(adata.obs['ground_truth'])]
            print(i + ' filter NA done')

        adata = construct_interaction(adata, n_neighbors=n_neighbors)
        print(i + ' build local graph done')
        adata_list.append(adata)
    print('load all slices done')

    return adata_list

def concatenate_slices(adata_list):

    highly_variable_genes_set = set(
        adata_list[0].var['highly_variable'][adata_list[0].var['highly_variable']].index
    )


    for adata in adata_list[1:]:
        current_set = set(adata.var['highly_variable'][adata.var['highly_variable']].index)
        highly_variable_genes_set = highly_variable_genes_set.intersection(current_set)


    adata = concat(adata_list, join='outer', label='batch', keys=[f'slice{i}' for i in range(len(adata_list))])


    adata_Vars = adata[:, adata.var.index.isin(highly_variable_genes_set)]
    if isinstance(adata_Vars.X, (csc_matrix, csr_matrix)):
        feat = adata_Vars.X.toarray()
    else:
        feat = adata_Vars.X


    adata.obsm['feat'] = feat

    print('merge done')
    return adata

from scipy.linalg import block_diag

def construct_whole_graph(adata_list, merged_adata):
    # 拼接所有切片的局部图邻接矩阵
    matrix_list = [i.obsm['local_graph'] for i in adata_list]
    adjacency = block_diag(*matrix_list)
    merged_adata.obsm['graph_neigh'] = adjacency

    print('whole graph constructed')
    return merged_adata


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['graph_neigh'] = adj

def construct_interaction_KNN(adata, n_neighbors=4):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj
    print('Graph constructed!')


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'









