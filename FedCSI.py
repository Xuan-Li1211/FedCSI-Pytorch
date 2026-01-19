import argparse
import copy
import os
import random
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch.nn import Module
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import community as community_louvain
from torch_scatter import scatter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# --- 1. 通用工具函数 (无修改) ---
def set_seed(seed):
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
    min_size, min_require_size = 0, 10
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
    communicate_indexes, in_com_train_data_indexes, edge_indexes_clients = [], [], []
    for i in tqdm(range(num_clients), desc="构建客户端子图"):
        client_nodes = torch.tensor(split_data_indexes[i])
        subset, current_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=client_nodes, num_hops=L_hop, edge_index=edge_index,
            relabel_nodes=True, num_nodes=num_nodes
        )
        communicate_indexes.append(subset)
        edge_indexes_clients.append(current_edge_index)

        inter = intersect1d(client_nodes.cpu(), idx_train.cpu())
        if subset.numel() > 0 and inter.numel() > 0:
            subset_map = {node_id.item(): local_id for local_id, node_id in enumerate(subset)}
            local_indices = [subset_map[node_id.item()] for node_id in inter if node_id.item() in subset_map]
            in_com_train_data_indexes.append(torch.tensor(local_indices, dtype=torch.long))
        else:
            in_com_train_data_indexes.append(torch.tensor([], dtype=torch.long))

    return communicate_indexes, in_com_train_data_indexes, edge_indexes_clients


# --- 2. 客户端画像与聚类功能 (无修改) ---
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

    best_score = -1.1
    best_labels, best_method = None, "default"

    for name, algorithm in [
        ('KMeans', KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')),
        ('Spectral', SpectralClustering(n_clusters=num_clusters, random_state=42, affinity='rbf'))
    ]:
        try:
            labels = algorithm.fit_predict(scaled_profiles)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(scaled_profiles, labels)
                if score > best_score:
                    best_score, best_labels, best_method = score, labels, name
        except Exception:
            pass

    if best_labels is None:
        best_labels = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit_predict(scaled_profiles)

    print(f"聚类完成: 方法={best_method}, 轮廓系数={best_score:.4f}")
    return best_labels


def visualize_clusters(client_profiles, cluster_assignments, save_path='client_clusters.png'):
    print("正在生成客户端聚类可视化图...")
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(np.array(client_profiles))

    perplexity_value = max(min(5, len(client_profiles) - 1), 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, n_iter=1000)
    profiles_2d = tsne.fit_transform(scaled_profiles)

    plt.figure(figsize=(8, 6))
    num_clusters = len(np.unique(cluster_assignments))
    scatter = plt.scatter(profiles_2d[:, 0], profiles_2d[:, 1], c=cluster_assignments,
                          cmap=plt.get_cmap('viridis', num_clusters))

    if num_clusters > 1:
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'簇 {i}' for i in range(num_clusters)])
    plt.title('Client Profile Clusters (t-SNE Visualization)')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"客户端聚类可视化图已保存至: {save_path}")


# --- 3. 预训练专用模型和客户端 (无修改) ---
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(out_channels, out_channels))
        self.readout = nn.Linear(out_channels, in_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate,
                                          weight_decay=args.weight_decay)

    def pretrain_step(self, global_model_state):
        self.model.load_state_dict(global_model_state)
        self.model.train()
        for _ in range(self.args.local_step):
            self.optimizer.zero_grad()
            embeddings = self.model(self.features, self.edge_index)
            reconstructed = self.model.decode(embeddings)
            if self.train_idx.numel() > 0:
                loss = F.mse_loss(reconstructed[self.train_idx], self.features[self.train_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 添加梯度裁剪
                self.optimizer.step()
        return self.model.state_dict()

    @torch.no_grad()
    def generate_embeddings(self, model_state):
        self.model.load_state_dict(model_state)
        self.model.eval()
        return self.model(self.features, self.edge_index)


# --- 4. 主模型定义 (无修改) ---
class CrossScaleGNN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers, num_communities):
        super(CrossScaleGNN, self).__init__()
        self.node_convs = torch.nn.ModuleList()
        self.node_convs.append(GCNConv(nfeat, nhid, normalize=True, cached=False))
        for _ in range(num_layers - 1):
            self.node_convs.append(GCNConv(nhid, nhid, normalize=True, cached=False))
        self.comm_convs = torch.nn.ModuleList()
        self.comm_convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        for _ in range(num_layers - 1):
            self.comm_convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        self.gate_nn = nn.Linear(nhid * 2, 1)
        self.classifier = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, node_features, node_edge_index, comm_features, comm_edge_index, node_to_comm_map):
        h_node = node_features
        for conv in self.node_convs:
            h_node = conv(h_node, node_edge_index)
            h_node = F.relu(h_node)
            h_node = F.dropout(h_node, p=self.dropout, training=self.training)
        h_comm_expanded = comm_features.to(h_node.device)[node_to_comm_map]
        gate_input = torch.cat([h_node, h_comm_expanded], dim=1)
        w = torch.sigmoid(self.gate_nn(gate_input))
        h_final = w * h_node + (1 - w) * h_comm_expanded
        output = self.classifier(h_final)
        return torch.log_softmax(output, dim=-1)

    def run_comm_gnn(self, comm_features, comm_edge_index):
        h_comm = comm_features
        for conv in self.comm_convs:
            h_comm = conv(h_comm, comm_edge_index)
            h_comm = F.relu(h_comm)
        return h_comm


# --- 5. 分簇客户端与服务器定义 ---
# <<< MODIFIED: ClusteredFedCSIClient 的 train 方法签名被修改
class ClusteredFedCSIClient:
    def __init__(self, client_id, cluster_id, model, local_features, local_edge_index, local_labels,
                 local_train_indices, node_to_comm_map, args):
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.model = copy.deepcopy(model)
        self.features = local_features
        self.edge_index = local_edge_index
        self.labels = local_labels
        self.train_idx = local_train_indices
        self.node_to_comm_map = node_to_comm_map
        self.args = args
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate,
                                          weight_decay=args.weight_decay)

    def train(self, global_model_state, global_comm_reps, global_comm_edge_index, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.model.load_state_dict(global_model_state)
        self.model.train()
        total_loss = 0
        steps_with_loss = 0
        for _ in range(self.args.local_step):
            self.optimizer.zero_grad()
            if self.train_idx.numel() > 0:
                output = self.model(self.features, self.edge_index, global_comm_reps, global_comm_edge_index, self.node_to_comm_map)
                loss = F.nll_loss(output[self.train_idx], self.labels[self.train_idx])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                steps_with_loss += 1
        avg_loss = total_loss / steps_with_loss if steps_with_loss > 0 else 0
        return self.model.state_dict(), avg_loss

    @torch.no_grad()
    def generate_initial_reps(self, global_model_state):
        self.model.load_state_dict(global_model_state)
        self.model.eval()
        reps = F.relu(self.model.node_convs[0](self.features, self.edge_index))
        return reps


# <<< MODIFIED: ClusteredServer 增加了优化器和调度器的管理
class ClusteredServer:
    def __init__(self, num_clusters, model_class, dataset, args, device):
        self.num_clusters = num_clusters
        self.cluster_models = {}
        self.clients_by_cluster = defaultdict(list)
        self.device = device
        self.args = args
        self.model_class = model_class
        self.dataset = dataset
        self.cluster_optimizers = {}
        self.cluster_schedulers = {}
        self.previous_comm_reps = {}

    def initialize_cluster_models(self, num_communities):
        for k in range(self.num_clusters):
            model = self.model_class(
                nfeat=self.dataset.num_features, nhid=self.args.num_hidden, nclass=self.dataset.num_classes,
                dropout=self.args.dropout, num_layers=self.args.num_layers, num_communities=num_communities
            ).to(self.device)
            self.cluster_models[k] = model
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-4, verbose=False
            )
            self.cluster_optimizers[k] = optimizer
            self.cluster_schedulers[k] = scheduler

    def register_client(self, client):
        self.clients_by_cluster[client.cluster_id].append(client)

    def train_cluster(self, cluster_id, comm_edge_index, node_to_comm_map, communicate_indexes, full_data_num_nodes):
        clients = self.clients_by_cluster.get(cluster_id, [])
        if not clients: return None, 0.0

        model = self.cluster_models[cluster_id]
        current_lr = self.cluster_optimizers[cluster_id].param_groups[0]['lr']

        all_reps, all_ids = [], []
        for client in clients:
            local_reps = client.generate_initial_reps(model.state_dict())
            all_reps.append(local_reps)
            all_ids.append(communicate_indexes[client.client_id])
        all_reps_cat = torch.cat(all_reps, dim=0)
        all_ids_cat = torch.cat(all_ids, dim=0).to(self.device)
        initial_node_reps = scatter(src=all_reps_cat, index=all_ids_cat, dim=0, reduce="mean",
                                    dim_size=full_data_num_nodes)

        with torch.no_grad():
            num_communities = node_to_comm_map.max().item() + 1
            comm_features = scatter(src=initial_node_reps, index=node_to_comm_map, dim=0, reduce="mean",
                                    dim_size=num_communities)
            if self.previous_comm_reps.get(cluster_id) is not None:
                comm_features = 0.9 * comm_features + 0.1 * self.previous_comm_reps[cluster_id]
            self.previous_comm_reps[cluster_id] = comm_features.detach().clone()
            global_comm_reps = model.run_comm_gnn(comm_features, comm_edge_index)

        client_updates = [
            client.train(model.state_dict(), global_comm_reps, comm_edge_index, new_lr=current_lr)
            for client in clients
        ]

        client_model_states = [update[0] for update in client_updates]
        client_losses = [update[1] for update in client_updates]

        client_train_sizes = [len(client.train_idx) for client in clients]
        total_train_size = sum(client_train_sizes)
        weighted_avg_loss = sum(loss * size for loss, size in zip(client_losses,
                                                                  client_train_sizes)) / total_train_size if total_train_size > 0 else 0

        new_global_state = self.server_aggregate_weighted(model.state_dict(), client_model_states, clients)
        model.load_state_dict(new_global_state)
        return global_comm_reps, weighted_avg_loss

    def fuse_cluster_models(self, alpha: float):
        """
        在所有簇模型之间进行模型融合（知识共享）。
        每个簇的模型参数都会向“平均模型”移动一小步。
        alpha: 融合系数，代表向平均模型靠近的程度。
        """
        if len(self.cluster_models) <= 1:
            # 如果只有一个簇或没有簇，则无需融合
            return

        # 1. 计算所有簇模型的“平均模型”状态
        mean_state_dict = copy.deepcopy(self.cluster_models[0].state_dict())
        for key in mean_state_dict:
            mean_state_dict[key].zero_()

        for k in self.cluster_models.keys():
            model_state = self.cluster_models[k].state_dict()
            for key in mean_state_dict:
                mean_state_dict[key] += model_state[key]

        # 除以簇的数量得到平均值
        num_clusters = len(self.cluster_models)
        for key in mean_state_dict:
            mean_state_dict[key] /= num_clusters

        # 2. 让每个簇的模型向“平均模型”靠近 alpha 比例
        for k in self.cluster_models.keys():
            model_k_state = self.cluster_models[k].state_dict()
            for key in model_k_state:
                model_k_state[key] = (1.0 - alpha) * model_k_state[key] + alpha * mean_state_dict[key]

            self.cluster_models[k].load_state_dict(model_k_state)

    @staticmethod
    @torch.no_grad()
    def server_aggregate(global_model_state, client_model_states):
        aggregated_state_dict = copy.deepcopy(global_model_state)
        if not client_model_states: return aggregated_state_dict
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key].zero_()
            for state in client_model_states:
                aggregated_state_dict[key] += state[key]
            aggregated_state_dict[key] /= len(client_model_states)
        return aggregated_state_dict

    @torch.no_grad()
    def server_aggregate_weighted(self, global_model_state, client_model_states, clients):
        weights = [len(client.train_idx) for client in clients]
        total_weight = sum(weights)
        if total_weight == 0: return global_model_state
        normalized_weights = [w / total_weight for w in weights]
        aggregated_state_dict = copy.deepcopy(global_model_state)
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key].zero_()
            for i, state in enumerate(client_model_states):
                aggregated_state_dict[key] += normalized_weights[i] * state[key]
        return aggregated_state_dict


# --- 6. 评估函数 (无修改) ---
@torch.no_grad()
def evaluate_clustered_model(server: ClusteredServer, full_data, mask, comm_edge_index, node_to_comm_map,
                             comm_info_for_eval):
    all_preds = []
    cluster_weights = []
    num_clusters = server.num_clusters
    device = full_data.x.device

    for k in range(num_clusters):
        if k not in server.clients_by_cluster or not server.clients_by_cluster[k]: continue

        model = server.cluster_models[k]
        model.eval()

        comm_reps_k = comm_info_for_eval.get(k)
        if comm_reps_k is None: continue

        try:
            device = next(model.parameters()).device
            out = model(full_data.x.to(device),
                        full_data.edge_index.to(device),
                        comm_reps_k,
                        comm_edge_index,
                        node_to_comm_map)
            all_preds.append(torch.exp(out[mask]))

            clients_in_cluster = server.clients_by_cluster.get(k, [])
            cluster_weight = sum(len(client.train_idx) for client in clients_in_cluster)
            cluster_weights.append(cluster_weight)
        except Exception as e:
            print(f"评估簇 {k} 时出错: {e}")
            continue

    if not all_preds: return 0.0

    total_weight = sum(cluster_weights)
    if total_weight == 0: return 0.0

    weights = torch.tensor([w / total_weight for w in cluster_weights], device=all_preds[0].device).view(-1, 1, 1)
    all_preds_tensor = torch.stack(all_preds, dim=0)
    weighted_preds = weights * all_preds_tensor
    avg_preds = weighted_preds.sum(dim=0)
    final_pred = avg_preds.argmax(dim=1)

    acc = (final_pred.to(full_data.y.device) == full_data.y[mask]).sum().item() / mask.sum().item()
    return acc


# --- 7. 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="cora", type=str)
    parser.add_argument("-c", "--global_rounds", default=200, type=int)
    parser.add_argument("-pc", "--pretrain_rounds", default=150, type=int)
    parser.add_argument("-i", "--local_step", default=5, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.0009, type=float)
    parser.add_argument("-wd", "--weight_decay", default=5e-4, type=float)
    parser.add_argument("-n", "--n_trainer", default=10, type=int)
    parser.add_argument("-nl", "--num_layers", default=2, type=int)
    parser.add_argument("-nhop", "--num_hops", default=2, type=int)
    parser.add_argument("-nhid", "--num_hidden", default=128, type=int)
    parser.add_argument("-do", "--dropout", default=0.404, type=float)
    parser.add_argument("-p", "--patience", default=20, type=int)
    parser.add_argument("-g", "--gpu", action="store_true", default=True)
    parser.add_argument("-b", "--iid_beta", default=1, type=float)
    parser.add_argument("-s", "--seed", default=555, type=int)
    parser.add_argument("-k", "--num_clusters", default=5, type=int)
    parser.add_argument("--fusion_alpha", default=0.01, type=float)

    args = parser.parse_args()
    print(f"实验参数: {args}")

    set_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
    dataset = Planetoid(path, args.dataset)
    full_data = dataset[0]

    train_indices = np.where(full_data.train_mask.cpu().numpy())[0]
    train_labels = full_data.y.cpu().numpy()[train_indices]
    client_private_indices = label_dirichlet_partition(train_labels, args.n_trainer, args.iid_beta)

    global_edge_index = full_data.edge_index

    print("正在构建客户端子图...")
    communicate_indexes, in_com_train_indices, edge_indexes_clients = get_in_comm_indexes(
        global_edge_index, client_private_indices, args.n_trainer,
        args.num_hops, torch.from_numpy(train_indices), num_nodes=full_data.num_nodes
    )
    print("客户端子图构建完成。")

    # === 阶段 0 - 联邦预训练 ===
    print("\n=== 阶段 0: 执行联邦自监督预训练 ===")
    pretrain_encoder = GCNEncoder(
        in_channels=dataset.num_features,
        out_channels=args.num_hidden
    ).to(device)
    pretrain_clients = [
        PretrainingClient(
            model=pretrain_encoder,
            local_features=full_data.x[communicate_indexes[i]].to(device),
            local_edge_index=edge_indexes_clients[i].to(device),
            local_train_indices=in_com_train_indices[i].to(device),
            args=args
        ) for i in range(args.n_trainer)
    ]
    for round_idx in tqdm(range(args.pretrain_rounds), desc="联邦预训练"):
        client_model_states = [client.pretrain_step(pretrain_encoder.state_dict()) for client in pretrain_clients]
        new_state = ClusteredServer.server_aggregate(pretrain_encoder.state_dict(), client_model_states)
        pretrain_encoder.load_state_dict(new_state)
    print("联邦预训练完成。")

    # === 阶段 1 - 聚类 ===
    print("\n=== 阶段 1: 生成客户端画像并执行聚类 ===")
    client_profiles_vectors = []
    with torch.no_grad():
        for i in tqdm(range(args.n_trainer), desc="生成客户端画像"):
            embeddings = pretrain_clients[i].generate_embeddings(pretrain_encoder.state_dict())
            profile_vector = extract_client_profile(
                embeddings=embeddings,
                edge_index=edge_indexes_clients[i],
                num_nodes=len(communicate_indexes[i])
            )
            client_profiles_vectors.append(profile_vector)
    cluster_assignments = robust_clustering(client_profiles_vectors, args.num_clusters)
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    print("客户端聚类完成。各簇客户端数量:")
    for k, count in zip(unique_clusters, counts):
        print(f"  簇 {k}: {count} 个客户端")
    visualize_clusters(client_profiles_vectors, cluster_assignments)

    # === 阶段 2：分簇联邦学习 ===
    print("\n=== 阶段 2: 初始化并执行分簇FedCSI训练 ===")
    full_data = full_data.to(device)
    G = to_networkx(full_data.cpu(), to_undirected=True)
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))
    node_to_comm_map = torch.tensor([partition.get(i, 0) for i in range(full_data.num_nodes)], dtype=torch.long).to(device)
    print(f"全局社区检测完成，发现 {num_communities} 个社区。")
    comm_graph = nx.Graph()
    comm_graph.add_nodes_from(range(num_communities))
    for u, v in G.edges():
        c_u, c_v = partition.get(u, -1), partition.get(v, -1)
        if c_u != -1 and c_v != -1 and c_u != c_v:
            if comm_graph.has_edge(c_u, c_v):
                comm_graph[c_u][c_v]['weight'] += 1
            else:
                comm_graph.add_edge(c_u, c_v, weight=1)
    comm_data = from_networkx(comm_graph)
    comm_edge_index = comm_data.edge_index.to(device)

    actual_num_clusters = len(unique_clusters)
    server = ClusteredServer(actual_num_clusters, CrossScaleGNN, dataset, args, device)
    server.initialize_cluster_models(num_communities)

    clients = []
    for i in range(args.n_trainer):
        cluster_id = int(cluster_assignments[i])
        client_model_template = server.cluster_models[cluster_id]
        client = ClusteredFedCSIClient(
            client_id=i, cluster_id=cluster_id, model=client_model_template,
            local_features=full_data.x[communicate_indexes[i]].to(device),
            local_edge_index=edge_indexes_clients[i].to(device),
            local_labels=full_data.y[communicate_indexes[i]].to(device),
            local_train_indices=in_com_train_indices[i].to(device),
            node_to_comm_map=node_to_comm_map[communicate_indexes[i]],
            args=args
        )
        clients.append(client)
        server.register_client(client)
    print("分簇服务器和客户端初始化完成。")

    best_val_accuracy = 0.0
    best_model_states = {k: None for k in range(actual_num_clusters)}
    patience_counter = 0

    # <<< NEW: 初始化字典以存储历史记录, 增加 'loss' 键 >>>
    training_history = {'round': [], 'val_acc': [], 'loss': []}

    pbar = tqdm(range(args.global_rounds), desc="全局训练回合")
    for round_idx in pbar:
        comm_info_for_eval = {}
        round_losses = []  # 用于收集本轮各簇的loss

        for k in range(actual_num_clusters):
            # <<< MODIFIED: 接收 loss >>>
            comm_reps_k, cluster_loss = server.train_cluster(
                cluster_id=k,
                comm_edge_index=comm_edge_index,
                node_to_comm_map=node_to_comm_map,
                communicate_indexes=communicate_indexes,
                full_data_num_nodes=full_data.num_nodes
            )
            if comm_reps_k is not None:
                comm_info_for_eval[k] = comm_reps_k
                round_losses.append(cluster_loss)

        if args.fusion_alpha > 0:
            server.fuse_cluster_models(alpha=args.fusion_alpha)

        val_acc = evaluate_clustered_model(server, full_data, full_data.val_mask, comm_edge_index, node_to_comm_map,
                                           comm_info_for_eval)

        avg_round_loss = np.mean(round_losses) if round_losses else 0

        # <<< NEW: 记录本轮数据, 包括loss >>>
        training_history['round'].append(round_idx + 1)
        training_history['val_acc'].append(val_acc)
        training_history['loss'].append(avg_round_loss)

        # 更新进度条以显示loss
        pbar.set_postfix({"Val Acc": f"{val_acc:.4f}", "Avg Loss": f"{avg_round_loss:.4f}"})

        for k in server.cluster_schedulers:
            server.cluster_schedulers[k].step(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            for k in range(actual_num_clusters):
                if k in server.cluster_models:
                    best_model_states[k] = copy.deepcopy(server.cluster_models[k].state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n早停触发：验证集准确率在 {args.patience} 个回合内未提升。")
            break

    print(f"\n--- 分簇FedCSI 联邦学习结束 ---")

    print("\n--- 加载在验证集上表现最好的模型进行最终测试 ---")
    if any(s is not None for s in best_model_states.values()):
        for k in range(actual_num_clusters):
            if best_model_states.get(k) is not None:
                server.cluster_models[k].load_state_dict(best_model_states[k])

        final_comm_info = {}
        with torch.no_grad():
            for k in range(actual_num_clusters):
                clients_in_cluster = server.clients_by_cluster.get(k, [])
                if not clients_in_cluster: continue
                model_k = server.cluster_models[k]
                model_k.eval()
                all_reps, all_ids = [], []
                for client in clients_in_cluster:
                    reps = client.generate_initial_reps(model_k.state_dict())
                    all_reps.append(reps)
                    all_ids.append(communicate_indexes[client.client_id])
                all_reps_cat = torch.cat(all_reps, dim=0)
                all_ids_cat = torch.cat(all_ids, dim=0).to(device)
                initial_node_reps = scatter(src=all_reps_cat, index=all_ids_cat, dim=0, reduce="mean",
                                            dim_size=full_data.num_nodes)
                comm_features = scatter(src=initial_node_reps, index=node_to_comm_map, dim=0, reduce="mean")
                comm_reps_k = model_k.run_comm_gnn(comm_features, comm_edge_index)
                final_comm_info[k] = comm_reps_k

        final_test_acc = evaluate_clustered_model(server, full_data, full_data.test_mask, comm_edge_index,
                                                  node_to_comm_map, final_comm_info)
        print(f"最终测试集准确率: {final_test_acc:.4f}")
        print("\n--- 正在将收敛数据保存至Excel文件 ---")

        print("\n--- 正在将收敛数据保存至Excel文件 ---")
        if training_history['round']:
            df_history = pd.DataFrame(training_history)
            df_history.rename(columns={
                'round': 'Epoch (X)',
                'val_acc': 'Validation Accuracy (Y)',
                'loss': 'Average Training Loss'  # 新增的列
            }, inplace=True)

            # 创建描述性的文件名
            excel_filename = f"convergence_{args.dataset}_k{args.num_clusters}_beta{args.iid_beta}.xlsx"

            try:
                df_history.to_excel(excel_filename, index=False, sheet_name='Convergence_Data')
                print(f"收敛数据已成功保存至: {excel_filename}")
            except Exception as e:
                print(f"保存到Excel时出错: {e}")
        else:
            print("没有训练历史记录可供保存。")
        # <<< 新增代码结束 >>>
    else:
        print("训练失败或没有客户端，未能产生最佳模型。")