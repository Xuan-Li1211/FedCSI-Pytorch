import argparse
import copy
import os
import random
import gc
import pickle
from typing import List
from collections import defaultdict
import torch_sparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce
import networkx as nx
import community as community_louvain
from torch_scatter import scatter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch_geometric.transforms as T

# --- 1. 通用工具函数 ---
def set_seed(seed):
    """设置全局随机种子以保证实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def intersect1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts > 1]


def label_dirichlet_partition(labels: np.array, n_parties: int, beta: float) -> list[np.ndarray]:
    K = labels.max() + 1
    min_size, min_require_size = 0, 50  # For Arxiv, ensure clients have a decent number of nodes
    N = len(labels)
    split_data_indexes = []
    while min_size < min_require_size:
        idx_batch: list[list[int]] = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(np.array(idx_batch[j]))
    return split_data_indexes


def get_in_comm_indexes(edge_index: torch.Tensor, split_data_indexes: list, num_clients: int, L_hop: int,
                        idx_train: torch.Tensor, num_nodes: int) -> tuple:
    communicate_indexes, in_com_train_indices, edge_indexes_clients = [], [], []
    edge_index_cpu = edge_index.cpu()
    for i in tqdm(range(num_clients), desc="构建客户端子图"):
        client_nodes = torch.tensor(split_data_indexes[i])
        subset, current_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=client_nodes, num_hops=L_hop, edge_index=edge_index_cpu,
            relabel_nodes=True, num_nodes=num_nodes
        )
        communicate_indexes.append(subset)
        # 修复 OOM 核心所在：将提取出的 COO edge_index 转换为 SparseTensor！
        current_sparse_adj = torch_sparse.SparseTensor(
            row=current_edge_index[0], col=current_edge_index[1],
            sparse_sizes=(len(subset), len(subset)),
        )
        edge_indexes_clients.append(current_sparse_adj)
        inter = intersect1d(client_nodes, idx_train.cpu())
        in_com_train_indices.append(torch.searchsorted(subset, inter))
    return communicate_indexes, in_com_train_indices, edge_indexes_clients


# NEW: 数据处理改进 - 特征归一化
def preprocess_features(data, train_mask):
    """仅使用训练集统计量对所有特征进行标准化，避免数据泄露"""
    train_features = data.x[train_mask]
    train_mean = train_features.mean(dim=0, keepdim=True)
    train_std = train_features.std(dim=0, keepdim=True) + 1e-8
    data.x = (data.x - train_mean) / train_std
    print("特征归一化完成。")
    return data


# --- 2. 客户端画像与聚类功能 ---
def extract_client_profile(embeddings: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    with torch.no_grad():
        embedding_mean = embeddings.mean(dim=0).cpu().numpy()
        avg_degree = edge_index.size(1) * 2 / num_nodes if num_nodes > 0 else 0
        profile = np.concatenate([embedding_mean, np.array([num_nodes, avg_degree])]).astype(np.float32)
    return profile


def robust_clustering(client_profiles: List[np.ndarray], num_clusters: int) -> np.ndarray:
    profiles_array = np.array(client_profiles)
    if len(profiles_array) < num_clusters: return np.arange(len(profiles_array))
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(profiles_array)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaled_profiles)
    score = silhouette_score(scaled_profiles, labels) if len(np.unique(labels)) > 1 else -1
    print(f"聚类完成: 方法=KMeans, 轮廓系数={score:.4f}")
    return labels


def visualize_clusters(client_profiles, cluster_assignments, save_path='arxiv_client_clusters.png'):
    print("正在生成客户端聚类可视化图...")
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(np.array(client_profiles))
    perplexity_value = max(min(5, len(client_profiles) - 1), 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, n_iter=1000)
    profiles_2d = tsne.fit_transform(scaled_profiles)
    plt.figure(figsize=(10, 8))
    num_clusters = len(np.unique(cluster_assignments))
    scatter = plt.scatter(profiles_2d[:, 0], profiles_2d[:, 1], c=cluster_assignments,
                          cmap=plt.get_cmap('viridis', num_clusters), s=50, alpha=0.7)
    if num_clusters > 1:
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(num_clusters)])
    plt.title('Client Profile Clusters on OGBN-Arxiv (t-SNE Visualization)')
    plt.xlabel('t-SNE Feature 1');
    plt.ylabel('t-SNE Feature 2');
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight');
    plt.close()
    print(f"客户端聚类可视化图已保存至: {save_path}")


# --- 3. 预训练专用模型和客户端 (改进版) ---
class ImprovedGCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(out_channels, out_channels))
            self.bns.append(nn.BatchNorm1d(out_channels))
        self.readout = nn.Linear(out_channels, in_channels)

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i > 0: x = x + x_prev  # 残差连接
        return x

    def decode(self, x):
        return self.readout(x)


class PretrainingClient:
    def __init__(self, model, local_features, local_edge_index, local_train_indices, args):
        self.model = copy.deepcopy(model)
        self.features = local_features
        self.edge_index = local_edge_index
        self.train_idx = local_train_indices
        self.args = args
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate,
                                           weight_decay=args.weight_decay)

    def pretrain_step(self, global_model_state):
        self.model.load_state_dict(global_model_state)
        self.model.train()
        for _ in range(self.args.local_step):
            self.optimizer.zero_grad()
            edge_index_on_device = self.edge_index.to(self.features.device)
            embeddings = self.model(self.features, edge_index_on_device)
            reconstructed = self.model.decode(embeddings)
            if self.train_idx.numel() > 0:
                loss = F.mse_loss(reconstructed[self.train_idx], self.features[self.train_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
        return self.model.state_dict()

    @torch.no_grad()
    def generate_embeddings(self, model_state):
        self.model.load_state_dict(model_state)
        self.model.eval()
        edge_index_on_device = self.edge_index.to(self.features.device)
        return self.model(self.features, edge_index_on_device)


# --- 4. 主模型定义 (改进版) ---
class ImprovedCrossScaleGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers, num_communities):
        super().__init__()
        # 节点级GNN
        self.node_convs = nn.ModuleList()
        self.node_bns = nn.ModuleList()
        self.node_convs.append(GCNConv(nfeat, nhid))
        self.node_bns.append(nn.BatchNorm1d(nhid))
        for _ in range(num_layers - 1):
            self.node_convs.append(GCNConv(nhid, nhid))
            self.node_bns.append(nn.BatchNorm1d(nhid))

        # 社区级GNN
        self.comm_convs = nn.ModuleList()
        self.comm_bns = nn.ModuleList()
        self.comm_convs.append(GCNConv(nhid, nhid))
        self.comm_bns.append(nn.BatchNorm1d(nhid))
        for _ in range(num_layers - 1):
            self.comm_convs.append(GCNConv(nhid, nhid))
            self.comm_bns.append(nn.BatchNorm1d(nhid))

        # 注意力门控
        self.attention_gate = nn.MultiheadAttention(nhid, num_heads=8, dropout=dropout, batch_first=True)
        self.gate_norm = nn.LayerNorm(nhid)

        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(nhid, nhid // 2),
            nn.BatchNorm1d(nhid // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(nhid // 2, nclass)
        )
        self.dropout = dropout

    def forward(self, node_features, node_edge_index, comm_features, comm_edge_index, node_to_comm_map):
        h_node = node_features
        for i, (conv, bn) in enumerate(zip(self.node_convs, self.node_bns)):
            h_prev = h_node
            h_node = F.dropout(F.relu(bn(conv(h_node, node_edge_index))), p=self.dropout, training=self.training)
            if i > 0: h_node = h_node + h_prev

        h_comm = self.run_comm_gnn(comm_features, comm_edge_index)

        # 确保社区映射的有效性和维度匹配
        if node_to_comm_map is not None:
            # 确保索引在有效范围内
            max_comm_id = h_comm.size(0) - 1
            safe_comm_map = torch.clamp(node_to_comm_map, 0, max_comm_id)
            h_comm_expanded = h_comm[safe_comm_map]

            # 多头注意力融合
            combined_features = torch.stack([h_node, h_comm_expanded], dim=1)  # Shape: [N, 2, D]
            attended_features, _ = self.attention_gate(combined_features, combined_features, combined_features)
            h_final = self.gate_norm(attended_features.mean(dim=1) + h_node)  # 残差连接
        else:
            # 如果没有社区映射，只使用节点特征
            h_final = h_node

        return F.log_softmax(self.classifier(h_final), dim=-1)

    def run_comm_gnn(self, comm_features, comm_edge_index):
        h_comm = comm_features
        for i, (conv, bn) in enumerate(zip(self.comm_convs, self.comm_bns)):
            h_prev = h_comm
            h_comm = F.dropout(F.relu(bn(conv(h_comm, comm_edge_index))), p=self.dropout, training=self.training)
            if i > 0: h_comm = h_comm + h_prev
        return h_comm


# --- 5. 分簇客户端与服务器定义 (改进版 + 数据泄露修复) ---
class ImprovedClusteredFedCSIClient:
    def __init__(self, client_id, cluster_id, model, local_features, local_edge_index, local_labels,
                 local_train_indices, global_train_mask_in_subgraph, args):
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.model = copy.deepcopy(model)
        self.features = local_features
        self.edge_index = local_edge_index
        self.labels = local_labels
        self.train_idx = local_train_indices
        self.global_train_mask = global_train_mask_in_subgraph  # For fixing data leak
        self.args = args
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate,
                                           weight_decay=args.weight_decay)

    def train(self, global_model_state, global_comm_reps, global_comm_edge_index, node_to_comm_map, new_lr):
        for param_group in self.optimizer.param_groups: param_group['lr'] = new_lr
        self.model.load_state_dict(global_model_state)
        self.model.train()
        for _ in range(self.args.local_step):
            self.optimizer.zero_grad()
            if self.train_idx.numel() > 0:
                output = self.model(self.features, self.edge_index, global_comm_reps, global_comm_edge_index,
                                    node_to_comm_map)
                loss = self._compute_loss_with_smoothing(output[self.train_idx], self.labels[self.train_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
        return self.model.state_dict()

    def _compute_loss_with_smoothing(self, output, labels, smoothing=0.1):
        num_classes = output.size(-1)
        smooth_labels = torch.full_like(output, smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        return -(smooth_labels * output).sum(dim=-1).mean()

    @torch.no_grad()
    def generate_initial_reps(self, global_model_state):
        """数据泄露修复: 只为训练节点生成初始表示"""
        self.model.load_state_dict(global_model_state)
        self.model.eval()
        reps = F.relu(self.model.node_convs[0](self.features, self.edge_index))
        return reps[self.global_train_mask], self.global_train_mask


class ImprovedClusteredServer:
    def __init__(self, num_clusters, model_class, dataset, args, device):
        self.num_clusters = num_clusters
        self.cluster_models = {k: model_class(
            nfeat=dataset.num_features, nhid=args.num_hidden, nclass=dataset.num_classes,
            dropout=args.dropout, num_layers=args.num_layers, num_communities=1  # Placeholder
        ).to(device) for k in range(num_clusters)}
        self.clients_by_cluster = defaultdict(list)
        self.device = device;
        self.args = args
        self.cluster_optimizers = {
            k: torch.optim.AdamW(m.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            for k, m in self.cluster_models.items()}
        self.cluster_schedulers = {
            k: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=10, T_mult=2, eta_min=1e-6)
            for k, o in self.cluster_optimizers.items()}

    def update_num_communities(self, num_communities):
        for k in self.cluster_models:
            self.cluster_models[k] = type(self.cluster_models[k])(
                nfeat=self.cluster_models[k].node_convs[0].in_channels,
                nhid=self.args.num_hidden, nclass=self.cluster_models[k].classifier[-1].out_features,
                dropout=self.args.dropout, num_layers=len(self.cluster_models[k].node_convs),
                num_communities=num_communities
            ).to(self.device)
            # Re-initialize optimizer and scheduler for the new model
            self.cluster_optimizers[k] = torch.optim.AdamW(self.cluster_models[k].parameters(),
                                                           lr=self.args.learning_rate,
                                                           weight_decay=self.args.weight_decay)
            self.cluster_schedulers[k] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.cluster_optimizers[k], T_0=10, T_mult=2, eta_min=1e-6)

    def register_client(self, client):
        self.clients_by_cluster[client.cluster_id].append(client)

    def server_aggregate(global_model_state, client_model_states):
        """
        Performs simple (unweighted) aggregation of model states.
        Defined as a static method to be easily called during pre-training.
        """
        aggregated_state_dict = copy.deepcopy(global_model_state)
        if not client_model_states:
            return aggregated_state_dict

        for key in aggregated_state_dict.keys():
            # Skip non-tensor buffers like num_batches_tracked
            if not aggregated_state_dict[key].is_floating_point():
                continue

            aggregated_state_dict[key].zero_()
            for state in client_model_states:
                aggregated_state_dict[key] += state[key]

            aggregated_state_dict[key] /= len(client_model_states)

        return aggregated_state_dict

    def train_cluster(self, cluster_id, comm_edge_index, node_to_comm_map_global, full_data):
        clients = self.clients_by_cluster.get(cluster_id, [])
        if not clients: return None
        model = self.cluster_models[cluster_id]
        optimizer = self.cluster_optimizers[cluster_id]

        # 数据泄露修复: 只收集训练节点的表示
        all_train_reps, all_train_global_ids = [], []
        for client in clients:
            train_reps, train_mask_in_subgraph = client.generate_initial_reps(model.state_dict())
            global_ids_in_subgraph = full_data.communicate_indexes[client.client_id]
            train_global_ids = global_ids_in_subgraph[train_mask_in_subgraph.cpu()]
            all_train_reps.append(train_reps)
            all_train_global_ids.append(train_global_ids)

        if not all_train_reps:  # 如果没有训练数据，跳过
            return None

        all_train_reps_cat = torch.cat(all_train_reps, dim=0)
        all_train_global_ids_cat = torch.cat(all_train_global_ids, dim=0).to(self.device)

        # 修复索引越界问题
        with torch.no_grad():
            # 确保索引在有效范围内
            max_node_id = node_to_comm_map_global.size(0) - 1
            valid_indices = all_train_global_ids_cat <= max_node_id

            if not valid_indices.all():
                print(f"警告: 发现越界索引，过滤后保留 {valid_indices.sum()}/{len(valid_indices)} 个节点")
                all_train_global_ids_cat = all_train_global_ids_cat[valid_indices]
                all_train_reps_cat = all_train_reps_cat[valid_indices]

            if all_train_global_ids_cat.numel() == 0:
                print(f"集群 {cluster_id} 没有有效的训练节点，跳过")
                return None

            # 数据泄露修复: 只用训练节点计算社区特征
            train_comm_map = node_to_comm_map_global[all_train_global_ids_cat]

            # 确保社区ID在有效范围内
            max_comm_id = comm_edge_index.max().item() if comm_edge_index.numel() > 0 else 0
            valid_comm_mask = train_comm_map <= max_comm_id

            if not valid_comm_mask.all():
                print(f"警告: 发现越界社区ID，过滤后保留 {valid_comm_mask.sum()}/{len(valid_comm_mask)} 个节点")
                train_comm_map = train_comm_map[valid_comm_mask]
                all_train_reps_cat = all_train_reps_cat[valid_comm_mask]

            if train_comm_map.numel() == 0:
                print(f"集群 {cluster_id} 没有有效的社区映射，跳过")
                return None

            comm_features = scatter(src=all_train_reps_cat, index=train_comm_map, dim=0, reduce="mean",
                                    dim_size=max_comm_id + 1)
            global_comm_reps = model.run_comm_gnn(comm_features, comm_edge_index)

        current_lr = optimizer.param_groups[0]['lr']

        # 为每个客户端准备局部的社区映射
        client_model_states = []
        for c in clients:
            # 获取客户端节点到全局节点的映射
            global_node_ids = full_data.communicate_indexes[c.client_id]
            # 获取这些全局节点对应的社区ID
            local_comm_map = node_to_comm_map_global[global_node_ids]

            client_state = c.train(model.state_dict(), global_comm_reps, comm_edge_index, local_comm_map, current_lr)
            client_model_states.append(client_state)

        # Weighted Aggregation
        weights = [len(c.train_idx) for c in clients]
        total_weight = sum(weights)
        if total_weight > 0:
            norm_weights = [w / total_weight for w in weights]
            new_state = model.state_dict()
            for key in new_state:
                if new_state[key].is_floating_point():
                    new_state[key].zero_()
                    for i, state in enumerate(client_model_states):
                        new_state[key] += state[key] * norm_weights[i]
            model.load_state_dict(new_state)

        return global_comm_reps

    def fuse_cluster_models(self, alpha: float):
        if len(self.cluster_models) <= 1: return
        mean_state = {k: v.cpu() for k, v in self.cluster_models[0].state_dict().items()}
        for i in range(1, self.num_clusters):
            for k, v in self.cluster_models[i].state_dict().items():
                mean_state[k] += v.cpu()
        for k in mean_state:
            if mean_state[k].is_floating_point(): mean_state[k] /= self.num_clusters

        for i in range(self.num_clusters):
            state = self.cluster_models[i].state_dict()
            for k in state:
                state[k] = (1.0 - alpha) * state[k] + alpha * mean_state[k].to(self.device)
            self.cluster_models[i].load_state_dict(state)


# --- 6. 评估函数 (改进版) ---
@torch.no_grad()
def evaluate_clustered_model(server, full_data, mask, comm_edge_index, node_to_comm_map, comm_info_for_eval):
    all_preds, cluster_weights = [], []
    for k in range(server.num_clusters):
        if k not in server.clients_by_cluster or not server.clients_by_cluster[k]: continue
        model = server.cluster_models[k]
        model.eval()
        comm_reps_k = comm_info_for_eval.get(k)
        if comm_reps_k is None: continue

        # 修复索引越界问题
        node_to_comm_map_for_model = node_to_comm_map
        if full_data.x.size(0) != node_to_comm_map.size(0):
            node_to_comm_map_for_model = node_to_comm_map[:full_data.x.size(0)]

        # 确保所有索引都在有效范围内
        max_comm_id = comm_reps_k.size(0) - 1
        valid_comm_mask = node_to_comm_map_for_model <= max_comm_id

        if not valid_comm_mask.all():
            # 对越界的社区ID设置为0（或其他有效值）
            node_to_comm_map_for_model = torch.clamp(node_to_comm_map_for_model, 0, max_comm_id)

        out = model(full_data.x, full_data.adj_t, comm_reps_k, comm_edge_index, node_to_comm_map_for_model)
        all_preds.append(torch.exp(out[mask]))
        cluster_weights.append(sum(len(c.train_idx) for c in server.clients_by_cluster[k]))

    if not all_preds: return 0.0
    total_weight = sum(cluster_weights)
    if total_weight == 0: return 0.0

    weights = torch.tensor([w / total_weight for w in cluster_weights], device=all_preds[0].device).view(-1, 1, 1)
    avg_preds = (weights * torch.stack(all_preds, dim=0)).sum(dim=0)
    final_pred = avg_preds.argmax(dim=1)

    correct = (final_pred == full_data.y.squeeze(1)[mask]).sum().item()
    return correct / mask.sum().item()


# --- 7. 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Federated Clustered GNN on OGBN-Arxiv")
    # 更新超参数
    parser.add_argument("-d", "--dataset", default="ogbn-arxiv", type=str)
    parser.add_argument("-c", "--global_rounds", default=150, type=int)
    parser.add_argument("-pc", "--pretrain_rounds", default=100, type=int)
    parser.add_argument("-i", "--local_step", default=5, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.002, type=float)
    parser.add_argument("-wd", "--weight_decay", default=1e-4, type=float)
    parser.add_argument("-n", "--n_trainer", default=10, type=int)
    parser.add_argument("-nl", "--num_layers", default=3, type=int)
    parser.add_argument("-nhop", "--num_hops", default=1, type=int)
    parser.add_argument("-nhid", "--num_hidden", default=128, type=int)
    parser.add_argument("-do", "--dropout", default=0.5, type=float)
    parser.add_argument("-p", "--patience", default=20, type=int)
    parser.add_argument("-g", "--gpu", action="store_true", default=True)
    parser.add_argument("-b", "--iid_beta", default=1, type=float)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-k", "--num_clusters", default=2, type=int)
    parser.add_argument("--fusion_alpha", default=0.05, type=float)
    args = parser.parse_args()
    print(f"实验参数: {args}")

    set_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 数据加载与预处理 ---
    print("正在加载 ogbn-arxiv 数据集...")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='dataset/',
                                     transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    # 创建Masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool);
    data.train_mask[train_idx] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool);
    data.val_mask[val_idx] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool);
    data.test_mask[test_idx] = True

    # 改进: 特征归一化
    data = preprocess_features(data, data.train_mask)

    # --- 联邦数据划分 ---
    train_labels = data.y.cpu().numpy().squeeze()[train_idx.numpy()]
    client_private_indices = label_dirichlet_partition(train_labels, args.n_trainer, args.iid_beta)

    row, col, _ = data.adj_t.coo()
    global_edge_index = torch.stack([row, col], dim=0)

    data.communicate_indexes, data.in_com_train_indices, data.edge_indexes_clients = get_in_comm_indexes(
        global_edge_index, client_private_indices, args.n_trainer,
        args.num_hops, train_idx, num_nodes=data.num_nodes
    )

    # --- 阶段 0: 联邦预训练 ---
    print("\n=== 阶段 0: 执行联邦自监督预训练 ===")
    pretrain_encoder = ImprovedGCNEncoder(dataset.num_features, args.num_hidden, num_layers=3).to(device)
    pretrain_clients = [PretrainingClient(
        pretrain_encoder, data.x[data.communicate_indexes[i]].to(device),
        data.edge_indexes_clients[i], data.in_com_train_indices[i].to(device), args
    ) for i in range(args.n_trainer)]

    for _ in tqdm(range(args.pretrain_rounds), desc="联邦预训练"):
        client_states = [c.pretrain_step(pretrain_encoder.state_dict()) for c in pretrain_clients]
        new_state = ImprovedClusteredServer.server_aggregate(pretrain_encoder.state_dict(), client_states)
        pretrain_encoder.load_state_dict(new_state)

    # --- 阶段 1: 客户端聚类 ---
    print("\n=== 阶段 1: 生成客户端画像并执行聚类 ===")
    client_profiles = [extract_client_profile(
        c.generate_embeddings(pretrain_encoder.state_dict()), c.edge_index, len(c.features)
    ) for c in tqdm(pretrain_clients, desc="生成客户端画像")]

    cluster_assignments = robust_clustering(client_profiles, args.num_clusters)
    visualize_clusters(client_profiles, cluster_assignments)

    # --- 阶段 2: 分簇联邦学习 ---
    print("\n=== 阶段 2: 初始化并执行分簇FedCSI训练 ===")
    # 社区检测
    partition_file = 'arxiv_community_partition.pkl'
    if os.path.exists(partition_file):
        with open(partition_file, 'rb') as f:
            full_partition = pickle.load(f)
        num_communities = len(set(full_partition.values()))
        print(f"从缓存加载社区划分，共 {num_communities} 个社区")
    else:
        print("正在执行社区检测...")
        G = to_networkx(data, to_undirected=True)
        full_partition = community_louvain.best_partition(G)
        num_communities = len(set(full_partition.values()))
        with open(partition_file, 'wb') as f:
            pickle.dump(full_partition, f)
        print(f"社区检测完成，共发现 {num_communities} 个社区")

    # 修复社区映射中的索引问题
    node_to_comm_map = torch.tensor([full_partition.get(i, 0) for i in range(data.num_nodes)], dtype=torch.long).to(
        device)

    # 确保社区ID连续且从0开始
    unique_comm_ids = torch.unique(node_to_comm_map)
    if unique_comm_ids.min() != 0 or unique_comm_ids.max() != len(unique_comm_ids) - 1:
        print("重新映射社区ID以确保连续性...")
        old_to_new = {old_id.item(): new_id for new_id, old_id in enumerate(unique_comm_ids)}
        node_to_comm_map = torch.tensor([old_to_new[comm_id.item()] for comm_id in node_to_comm_map],
                                        dtype=torch.long).to(device)
        num_communities = len(unique_comm_ids)
        print(f"重映射后共 {num_communities} 个社区")

    # 构建社区图
    print("正在构建社区间连接...")
    source_comms = node_to_comm_map[global_edge_index[0]]
    dest_comms = node_to_comm_map[global_edge_index[1]]
    inter_mask = source_comms != dest_comms

    if inter_mask.sum() > 0:
        comm_edge_index, comm_weights = coalesce(
            torch.stack([source_comms[inter_mask], dest_comms[inter_mask]], dim=0),
            torch.ones(inter_mask.sum().item(), device=device),
            num_nodes=num_communities, reduce="add"
        )
    else:
        # 如果没有社区间连接，创建空的边索引
        comm_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        print("警告: 没有发现社区间连接")

    # 初始化服务器和客户端
    actual_num_clusters = len(np.unique(cluster_assignments))
    server = ImprovedClusteredServer(actual_num_clusters, ImprovedCrossScaleGNN, dataset, args, device)
    server.update_num_communities(num_communities)

    clients = []
    for i in range(args.n_trainer):
        cluster_id = int(cluster_assignments[i])
        subgraph_nodes = data.communicate_indexes[i]

        # 数据泄露修复: 创建客户端独有的训练节点掩码
        global_train_mask_in_subgraph = torch.zeros(len(subgraph_nodes), dtype=torch.bool)
        subgraph_train_nodes_local_idx = data.in_com_train_indices[i]
        global_train_mask_in_subgraph[subgraph_train_nodes_local_idx] = True

        clients.append(ImprovedClusteredFedCSIClient(
            client_id=i, cluster_id=cluster_id, model=server.cluster_models[cluster_id],
            local_features=data.x[subgraph_nodes].to(device),
            # 【修复点】：删除 torch_geometric.utils.to_undirected，直接使用已经转好 SparseTensor 的数据
            local_edge_index=data.edge_indexes_clients[i].to(device),
            local_labels=data.y.squeeze(1)[subgraph_nodes].to(device),
            local_train_indices=subgraph_train_nodes_local_idx.to(device),
            global_train_mask_in_subgraph=global_train_mask_in_subgraph.to(device),
            args=args
        ))
        server.register_client(clients[-1])

    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    best_model_states = {k: None for k in range(actual_num_clusters)}


    # =====================================================================
    # === 新增模块：通信成本量化分析 (应对审稿人要求) ===
    # =====================================================================
    def get_model_size_mb(model_to_measure):
        """计算模型的可训练参数占用的总内存大小(MB)"""
        total_params = sum(p.numel() for p in model_to_measure.parameters() if p.requires_grad)
        return total_params * 4 / (1024 * 1024)  # Float32 类型占 4 Bytes


    # 1. 计算各个单体模型的大小
    pretrain_size = get_model_size_mb(pretrain_encoder)
    # 取任意一个集群的 CrossScaleGNN 作为代表
    cross_scale_gnn_size = get_model_size_mb(server.cluster_models[0])


    # 模拟构建一个基线的 FedGCN (具有相同层数和维度，但没有社区模块和注意力门控)
    class DummyFedGCN(nn.Module):
        def __init__(self, nfeat, nhid, nclass, num_layers):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(nfeat, nhid))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(nhid, nhid))
            self.classifier = nn.Sequential(
                nn.Linear(nhid, nhid // 2), nn.ReLU(),
                nn.Linear(nhid // 2, nclass)
            )


    dummy_fedgcn = DummyFedGCN(dataset.num_features, args.num_hidden, dataset.num_classes, args.num_layers)
    fedgcn_size = get_model_size_mb(dummy_fedgcn)

    # 2. 计算整个生命周期的总通信量 (以 MB 为单位)
    client_num = args.n_trainer
    # 上传+下发 = 每次通信乘 2

    # [基线 FedGCN]: 总轮数 * 客户端数 * 2 * 模型大小
    fedgcn_total_comm = args.global_rounds * client_num * 2 * fedgcn_size

    # [你的 FedCSI]: 三个阶段的累计
    # 阶段0: 预训练轮数 * 客户端数 * 2 * 预训练编码器大小
    stage0_comm = args.pretrain_rounds * client_num * 2 * pretrain_size
    # 阶段1: 画像上传 (每个客户端传一个 F+2 维度的向量)
    stage1_comm = client_num * (dataset.num_features + 2) * 4 / (1024 * 1024)
    # 阶段2: 分簇联邦主训练
    stage2_comm = args.global_rounds * client_num * 2 * cross_scale_gnn_size
    fedcsi_total_comm = stage0_comm + stage1_comm + stage2_comm
    # === 新增：计算 1-hop/2-hop 的拓扑交换通信量 ===
    # Ogbn-arxiv 的特征维度 F
    F_dim = dataset.num_features
    # 计算所有客户端额外拉取的边界节点总数
    total_boundary_nodes = 0
    for i in range(args.n_trainer):
        # 扩展后的子图节点数
        extended_nodes = len(data.communicate_indexes[i])
        # 原始分配的私有节点数
        private_nodes = len(client_private_indices[i])
        # 差值就是通过网络拉取的边界节点
        total_boundary_nodes += (extended_nodes - private_nodes)

    # L-hop 特征交换通信量 = 边界节点总数 * 特征维度 * 4 Bytes
    l_hop_comm_mb = (total_boundary_nodes * F_dim * 4) / (1024 * 1024)

    # === 估算 FGSSL 的通信量 (作为对比基线) ===
    # 假设 FGSSL 的主模型大小和 FedGCN 差不多
    # FGSSL 额外开销：每轮每个客户端还要传输 C 个类别的特征原型
    # 原型通信量 = 轮数 * 客户端数 * 2(上下行) * (类别数 * 特征维度 * 4 Bytes)
    fgssl_prototype_mb = (args.global_rounds * client_num * 2 * (dataset.num_classes * F_dim * 4)) / (1024 * 1024)
    fgssl_total_comm = fedgcn_total_comm + fgssl_prototype_mb

    print(f"🔹 [拓扑交换] 1-hop 边界节点总数: {total_boundary_nodes}")
    print(f"   ├─ L-hop 通信开销 (FedGCN & FedCSI 共有): {l_hop_comm_mb:.2f} MB")
    print(f"🚀 FGSSL 累积总通信量 (理论估算): {fgssl_total_comm:.2f} MB")
    print(f"   ├─ 主模型传输: {fedgcn_total_comm:.2f} MB")
    print(f"   └─ 原型持续传输: {fgssl_prototype_mb:.2f} MB")
    # 3. 打印精美的统计报告
    print("\n" + "=" * 50)
    print("📊 联邦学习通信成本量化分析 (Communication Cost)")
    print("=" * 50)
    print(f"🔹 [单体大小] 基线 FedGCN: {fedgcn_size:.4f} MB")
    print(f"🔹 [单体大小] FedCSI 预训练编码器: {pretrain_size:.4f} MB")
    print(f"🔹 [单体大小] FedCSI 主模型 (CrossScale): {cross_scale_gnn_size:.4f} MB")
    print("-" * 50)
    print(f"🚀 FedGCN 累积总通信量 ({args.global_rounds}轮): {fedgcn_total_comm:.2f} MB")
    print(f"🚀 FedCSI 累积总通信量: {fedcsi_total_comm:.2f} MB")
    print(f"   ├─ 阶段0 (预训练): {stage0_comm:.2f} MB")
    print(f"   ├─ 阶段1 (画像上传): {stage1_comm:.6f} MB (极小,可忽略)")
    print(f"   └─ 阶段2 (主训练): {stage2_comm:.2f} MB")

    if fedgcn_total_comm > 0:
        increase_ratio = ((fedcsi_total_comm - fedgcn_total_comm) / fedgcn_total_comm) * 100
        print(f"📈 FedCSI 相对通信量增加: +{increase_ratio:.2f}%")
    print("=" * 50 + "\n")
    # =====================================================================
    print(f"开始分簇联邦训练，共 {actual_num_clusters} 个集群...")
    for round_idx in tqdm(range(args.global_rounds), desc="全局训练回合"):
        comm_info = {}
        for k in range(actual_num_clusters):
            comm_reps = server.train_cluster(k, comm_edge_index, node_to_comm_map, data)
            if comm_reps is not None: comm_info[k] = comm_reps

        if args.fusion_alpha > 0 and (round_idx + 1) % 10 == 0:
            server.fuse_cluster_models(args.fusion_alpha)

        # 验证
        if (round_idx + 1) % 5 == 0:  # 每5轮验证一次
            val_acc = evaluate_clustered_model(server, data.to(device), data.val_mask.to(device), comm_edge_index,
                                               node_to_comm_map, comm_info)

            for k in server.cluster_schedulers:
                server.cluster_schedulers[k].step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                for k in range(actual_num_clusters):
                    if k in server.cluster_models:
                        best_model_states[k] = copy.deepcopy(server.cluster_models[k].state_dict())
                print(f"\n新最佳验证准确率 @ 回合 {round_idx + 1}: {best_val_acc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= args.patience // 5:  # 调整早停条件
                print(f"\n早停触发 @ 回合 {round_idx + 1}")
                break

    # 最终测试
    print("\n--- 最终测试 ---")
    if any(s is not None for s in best_model_states.values()):
        for k, state in best_model_states.items():
            if state and k in server.cluster_models:
                server.cluster_models[k].load_state_dict(state)

        # 重新计算最终的社区表示用于评估
        final_comm_info = {}
        with torch.no_grad():
            for k in range(actual_num_clusters):
                if k not in server.clients_by_cluster:
                    continue
                clients_in_cluster = server.clients_by_cluster[k]
                if not clients_in_cluster: continue
                model_k = server.cluster_models[k]

                all_train_reps, all_train_global_ids = [], []
                for client in clients_in_cluster:
                    train_reps, train_mask_in_subgraph = client.generate_initial_reps(model_k.state_dict())
                    if train_reps.numel() == 0:
                        continue
                    global_ids_in_subgraph = data.communicate_indexes[client.client_id]
                    train_global_ids = global_ids_in_subgraph[train_mask_in_subgraph.cpu()]
                    all_train_reps.append(train_reps)
                    all_train_global_ids.append(train_global_ids)

                if not all_train_reps:
                    continue

                all_train_reps_cat = torch.cat(all_train_reps, dim=0)
                all_train_global_ids_cat = torch.cat(all_train_global_ids, dim=0).to(device)

                # 确保索引有效
                max_node_id = node_to_comm_map.size(0) - 1
                valid_indices = all_train_global_ids_cat <= max_node_id
                if not valid_indices.all():
                    all_train_global_ids_cat = all_train_global_ids_cat[valid_indices]
                    all_train_reps_cat = all_train_reps_cat[valid_indices]

                if all_train_global_ids_cat.numel() > 0:
                    train_comm_map = node_to_comm_map[all_train_global_ids_cat]
                    comm_features = scatter(src=all_train_reps_cat, index=train_comm_map, dim=0, reduce="mean",
                                            dim_size=num_communities)
                    final_comm_info[k] = model_k.run_comm_gnn(comm_features, comm_edge_index)

        test_acc = evaluate_clustered_model(server, data.to(device), data.test_mask.to(device), comm_edge_index,
                                            node_to_comm_map, final_comm_info)
        print(f"最佳验证集准确率: {best_val_acc:.4f}")
        print(f"最终测试集准确率: {test_acc:.4f}")
    else:
        print("训练失败，未能产生最佳模型。")