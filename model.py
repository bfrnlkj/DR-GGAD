import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


def normalize_score_torch(ano_score: torch.Tensor):
    min_score = torch.min(ano_score)
    max_score = torch.max(ano_score)
    norm_score = (ano_score - min_score) / (max_score - min_score + 1e-8)
    return norm_score


def prototype_alignment_loss(residual_embed, labels, cluster_centers, margin=2):
    dists = torch.cdist(residual_embed, cluster_centers)
    min_dist, _ = torch.max(dists, dim=1)
    normal_mask = (labels == 0)
    abnormal_mask = (labels == 1)
    normal_loss = min_dist[normal_mask].mean()
    if abnormal_mask.sum() > 0:
        abnormal_dists = dists[abnormal_mask]
        hinge = F.relu(margin - abnormal_dists)
        abnormal_loss = hinge.mean()
    else:
        abnormal_loss = torch.tensor(0.0, device=residual_embed.device)
    return normal_loss + abnormal_loss


def max_message(feature, adj_matrix):
    adj_matrix = adj_matrix.to_dense()
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    message = torch.sum(sim_matrix, 1)
    message = message * r_inv
    return torch.mean((1 - message) ** 2), message

class DR(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_layers=2, dropout_rate=0, activation='ReLU', num_hops=4, alpha=0.8,
                 beta=1, k=1, **kwargs):
        super(DR, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.K = k
        self.alpha = alpha
        if num_layers == 0:
            return
        self.layers.append(nn.Linear(in_feats, h_feats))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(h_feats, h_feats))
        self.dropout = nn.Dropout(0.2) if 0.2 > 0 else nn.Identity()
        self.cluster_centers_dict = {}
        self.cluster_mlps = nn.ModuleDict()
        self._initialized_datasets = set()
        self.node_mlps = nn.Sequential(nn.Linear(in_feats, in_feats * 2), nn.Dropout(dropout_rate),
                                       nn.Linear(in_feats * 2, in_feats), nn.BatchNorm1d(in_feats))

    def compute_residual_prototypes(self, h, normal_idx, dataset_name):
        x_list = h.x_list
        residual_list = []
        first_element = x_list[0]
        for h_i in x_list[1:]:
            dif = h_i - first_element
            residual_list.append(dif)
        residual_embed = torch.hstack(residual_list)
        H_normal = residual_embed[normal_idx]
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(H_normal.cpu().detach().numpy())
        self.cluster_centers_dict[dataset_name] = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(
            residual_embed.device)
        if dataset_name not in self._initialized_datasets:
            self.cluster_mlps[dataset_name] = nn.Linear(self.in_feats * self.num_hops, self.h_feats * self.num_hops).to(
                h.x.device)
            self._initialized_datasets.add(dataset_name)

    def forward(self, h):
        x_list = h.x_list
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
        residual_list = []
        first_element = x_list[0]
        for h_i in x_list[1:]:
            dif = h_i - first_element
            residual_list.append(dif)
        residual_embed = torch.hstack(residual_list)
        node_embed = self.node_mlps(h.x)
        return residual_embed, node_embed

    def computer_loss(self, h, name):
        labels = h.ano_labels
        residual_embed, node_embed = self.forward(h)
        cluster_centers = self.cluster_mlps[name](self.cluster_centers_dict[name])
        proto_loss = prototype_alignment_loss(residual_embed, labels, cluster_centers, 1)
        art_loss, _ = max_message(node_embed, h.adj_ori)
        total_loss = proto_loss + art_loss
        return total_loss

    def tam_score_euclidean(self, adj, node_feat):
        adj = adj.coalesce()
        row, col = adj.indices()
        N = node_feat.size(0)
        dist = torch.norm(node_feat[row] - node_feat[col], p=2, dim=1)
        score_sum = torch.zeros(N, device=node_feat.device)
        degree = torch.zeros(N, device=node_feat.device)
        score_sum.index_add_(0, row, dist)
        degree.index_add_(0, row, torch.ones_like(dist))
        anomaly_score = score_sum / (degree + 1e-8)
        anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)
        return anomaly_score

    def get_all_projected_cluster_centers(self):
        projected = []
        for name, centers in self.cluster_centers_dict.items():
            proj_mlp = self.cluster_mlps[name]
            projected.append(proj_mlp(centers))
        return torch.cat(projected, dim=0)

    @torch.no_grad()
    def get_anomaly_score(self, h):
        residual_embed, node_embed = self.forward(h)
        cluster_centers = self.get_all_projected_cluster_centers()

        dists = torch.cdist(residual_embed, cluster_centers)
        score_dis = dists.max(dim=1)[0]
        score_dis = (score_dis - score_dis.min()) / (score_dis.max() - score_dis.min() + 1e-8)
        h.adj = h.adj.coalesce()
        mask = h.adj.indices()[0] != h.adj.indices()[1]
        h.adj = torch.sparse_coo_tensor(h.adj.indices()[:, mask], h.adj.values()[mask], h.adj.size(),
                                        device=h.adj.device)

        t_score_dis = self.tam_score_euclidean(h.adj, node_embed)

        _, tam_score_cos = max_message(node_embed, h.adj_ori)
        message_list = []
        message_list.append(torch.unsqueeze(tam_score_cos, 0))
        message_list = torch.mean(torch.cat(message_list), 0)
        t_score_cos = 1 - normalize_score_torch(message_list)

        total_score = (score_dis) * self.alpha + (t_score_cos + t_score_dis) * (
                1 - self.alpha)
        return total_score
