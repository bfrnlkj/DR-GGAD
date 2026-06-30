import json
import os
import random

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.random_projection import GaussianRandomProjection
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix


def test_eval(labels, probs):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = roc_auc_score(labels, probs)
        score['AUPRC'] = average_precision_score(labels, probs)
    return score


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def feat_alignment(X, edges, dims):
    edge_src, edge_dst = edges
    num_edges = len(edge_src)

    if X.shape[1] < dims:
        transformer = GaussianRandomProjection(n_components=256, random_state=0)
        X = transformer.fit_transform(X.cpu().numpy())
    # Proj
    pca = PCA(n_components=dims, random_state=0)
    X_transformed = pca.fit_transform(X)
    X_transformed = torch.FloatTensor(X_transformed)
    # normalize
    X_min, _ = torch.min(X_transformed, dim=0)
    X_max, _ = torch.max(X_transformed, dim=0)
    X_s = (X_transformed - X_min) / (X_max - X_min)

    smooth_coefficients = torch.zeros(X_transformed.shape[1])
    for k in range(X_transformed.shape[1]):
        differences = X_s[edge_src, k] - X_s[edge_dst, k]
        smooth_coefficients[k] = torch.sum(differences ** 2) / num_edges
    _, sorted_indices = torch.sort(smooth_coefficients)
    X_reordered = X_transformed[:, sorted_indices]
    return X_reordered


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class Dataset:
    def __init__(self, dims, name='cora', prefix='./dataset/'):
        self.shot_mask = None
        self.shot_idx = None
        self.graph = None
        self.x_list = None
        self.name = name

        preprocess_filename = f'{prefix}{name}_{dims}.npz'
        if os.path.exists(preprocess_filename):
            with np.load(preprocess_filename, allow_pickle=True) as f:
                data = f['data'].item()
                feat = f['feat']
        else:
            data = sio.loadmat(f"{prefix + name}.mat")
            adj = data['Network']
            feat = data['Attributes']
            adj_sp = sp.csr_matrix(adj)
            row, col = adj_sp.nonzero()
            edge_index = torch.tensor([row, col], dtype=torch.long)
            if name in ['Amazon', 'YelpChi', 'tolokers', 'tfinance']:
                feat = sp.lil_matrix(feat)
                feat = preprocess_features(feat)
            else:
                feat = sp.lil_matrix(feat).toarray()
            feat = torch.FloatTensor(feat)
            feat = feat_alignment(feat, edge_index, dims)

        adj = data['Network'] if 'Network' in data else data['A']
        if name in ['YelpChi', 'Facebook']:
            adj_norm = normalize_adj(adj)
        else:
            adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        if name in ['YelpChi', 'Facebook']:
            adj_ori = sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj_ori = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
        label = data['Label'] if ('Label' in data) else data['gnd']

        self.label = label
        self.adj_norm = adj_norm
        self.feat = feat
        self.adj_ori = adj_ori
        ano_labels = torch.tensor(np.squeeze(np.array(self.label)), dtype=torch.float)
        data = Data(x=torch.tensor(self.feat, dtype=torch.float), x_list=self.x_list, adj=self.adj_norm,
                    ano_labels=ano_labels, shot_idx=self.shot_idx, shot_mask=self.shot_mask, adj_ori=self.adj_ori)
        self.graph = data

    def propagated(self, k):
        x = torch.FloatTensor(self.feat).cuda()
        edge_index = self.graph.adj._indices()  # [2, E]
        num_nodes = self.feat.shape[0]
        mask = edge_index[0] != edge_index[1]
        edge_index_noloop = edge_index[:, mask]
        values_noloop = torch.ones(mask.sum(), device=edge_index.device)
        A = torch.sparse_coo_tensor(edge_index_noloop, values_noloop, (num_nodes, num_nodes)).to(torch.float32).cuda()
        self.graph.origin_adj = A
        h_list = [x]
        for _ in range(k):
            h_list.append(torch.spmm(self.adj_norm.cuda(), h_list[-1]))
        self.graph.x_list = h_list

def read_json(model, shot, json_dir):
    filename = f"{json_dir}/{model}_{shot}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")
                return None
    else:
        print(f"JSON file {filename} not found.")
        return None