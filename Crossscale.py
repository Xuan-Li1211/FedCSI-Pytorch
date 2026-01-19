import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
import torch_sparse
from torch.nn import Module
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import community as community_louvain
from torch_scatter import scatter


# --- 1. 通用工具函数 ---
def set_seed(seed):
    """设置全局随机种子以保证实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU.
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
    for i in range(num_clients):
        client_nodes = torch.tensor(split_data_indexes[i])
        subset, current_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=client_nodes, num_hops=L_hop, edge_index=edge_index,
            relabel_nodes=True, num_nodes=num_nodes
        )
        communicate_indexes.append(subset)
        current_edge_index = torch_sparse.SparseTensor(
            row=current_edge_index[0], col=current_edge_index[1],
            sparse_sizes=(len(subset), len(subset)),
        )
        edge_indexes_clients.append(current_edge_index)
        inter = intersect1d(client_nodes, idx_train)
        in_com_train_data_indexes.append(torch.searchsorted(subset, inter))
    return communicate_indexes, in_com_train_data_indexes, edge_indexes_clients


# --- 2. 模型定义 ---

# 模型 1: 标准 GCN (用于 FedGCN 基线)
class GCN(Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, num_layers: int):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=False))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=False))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_t: torch_sparse.SparseTensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


# 模型 2: 跨尺度交互 GNN (Fed-CSI 创新模型)
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

    def forward(self, node_features, node_adj, comm_features, comm_adj, node_to_comm_map):
        h_node = node_features
        for conv in self.node_convs:
            h_node = conv(h_node, node_adj)
            h_node = F.relu(h_node)
            h_node = F.dropout(h_node, p=self.dropout, training=self.training)
        h_comm_expanded = comm_features[node_to_comm_map]
        gate_input = torch.cat([h_node, h_comm_expanded], dim=1)
        w = torch.sigmoid(self.gate_nn(gate_input))
        h_final = w * h_node + (1 - w) * h_comm_expanded
        output = self.classifier(h_final)
        return torch.log_softmax(output, dim=-1)

    def run_comm_gnn(self, comm_features, comm_adj):
        h_comm = comm_features
        for conv in self.comm_convs:
            h_comm = conv(h_comm, comm_adj)
            h_comm = F.relu(h_comm)
        return h_comm


# --- 3. 客户端定义 ---

# 客户端 1: 用于 FedGCN
class FedGCNClient:
    def __init__(self, client_id, model, local_features, local_adj, local_labels, local_train_indices, args):
        self.model = copy.deepcopy(model)
        self.features = local_features
        self.adj = local_adj
        self.labels = local_labels
        self.train_idx = local_train_indices
        self.args = args  # --- 修正点 ---
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def train(self, global_model_state):
        self.model.load_state_dict(global_model_state)
        self.model.train()
        for _ in range(self.args.local_step):
            self.optimizer.zero_grad()
            output = self.model(self.features, self.adj)
            loss = F.nll_loss(output[self.train_idx], self.labels[self.train_idx])
            loss.backward()
            self.optimizer.step()
        return self.model.state_dict()

# 客户端 2: 用于 Fed-CSI
class FedCSIClient:
    def __init__(self, client_id, model, local_features, local_adj, local_labels, local_train_indices, node_to_comm_map, args):
        self.model = copy.deepcopy(model)
        self.features = local_features
        self.adj = local_adj
        self.labels = local_labels
        self.train_idx = local_train_indices
        self.node_to_comm_map = node_to_comm_map
        self.args = args  # --- 修正点 ---
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def train(self, global_model_state, global_comm_reps, global_comm_adj):
        self.model.load_state_dict(global_model_state)
        self.model.train()
        for _ in range(self.args.local_step):
            self.optimizer.zero_grad()
            output = self.model(
                self.features, self.adj,
                global_comm_reps, global_comm_adj,
                self.node_to_comm_map
            )
            loss = F.nll_loss(output[self.train_idx], self.labels[self.train_idx])
            loss.backward()
            self.optimizer.step()
        return self.model.state_dict()

    @torch.no_grad()
    def generate_initial_reps(self, global_model_state):
        self.model.load_state_dict(global_model_state)
        self.model.eval()
        reps = F.relu(self.model.node_convs[0](self.features, self.adj))
        return reps


# --- 4. 服务端与评估逻辑 ---
@torch.no_grad()
def server_aggregate(global_model_state, client_model_states, skip_keys=None):
    if skip_keys is None:
        skip_keys = []
    aggregated_state_dict = copy.deepcopy(global_model_state)
    for key in aggregated_state_dict.keys():
        if any(skip_key in key for skip_key in skip_keys):
            continue
        aggregated_state_dict[key].zero_()
        for state in client_model_states:
            aggregated_state_dict[key] += state[key]
        aggregated_state_dict[key] /= len(client_model_states)
    return aggregated_state_dict

def evaluate_model(model, full_data, mask, comm_info=None):
    model.eval()

    if comm_info: # Fed-CSI
        global_comm_reps, comm_adj, node_to_comm_map = comm_info
        out = model(full_data.x, full_data.adj_t, global_comm_reps, comm_adj, node_to_comm_map)
    else: # FedGCN
        out = model(full_data.x, full_data.adj_t)

    pred = out.argmax(dim=1)
    correct = pred[mask] == full_data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc


# --- 5. 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="FedCSI", type=str, help="模型选择: FedGCN, FedCSI")
    parser.add_argument("-d", "--dataset", default="cora", type=str, help="数据集: cora, citeseer, pubmed")
    parser.add_argument("-c", "--global_rounds", default=200, type=int, help="全局联邦回合数")
    parser.add_argument("-i", "--local_step", default=5, type=int, help="客户端本地训练步数")
    parser.add_argument("-lr", "--learning_rate", default=0.002, type=float, help="学习率 (Adam推荐更小的值)")
    parser.add_argument("-wd", "--weight_decay", default=5e-3, type=float, help="权重衰减")
    parser.add_argument("-n", "--n_trainer", default=10, type=int, help="客户端数量")
    parser.add_argument("-nl", "--num_layers", default=2, type=int, help="GCN模型层数")
    parser.add_argument("-nhop", "--num_hops", default=2, type=int, help="L-hop邻居跳数")
    parser.add_argument("-nhid", "--num_hidden", default=256, type=int, help="隐藏层维度")
    parser.add_argument("-do", "--dropout", default=0.4, type=float, help="Dropout率")
    parser.add_argument("-p", "--patience", default=20, type=int, help="早停的耐心值")
    parser.add_argument("-g", "--gpu", action="store_true", default=True, help="如果设置，则使用GPU")
    parser.add_argument("-b", "--iid_beta", default=1, type=float, help="狄利克雷分布beta值 (越小越Non-IID)")
    parser.add_argument("-s", "--seed", default=42, type=int, help="随机种子")
    args = parser.parse_args()
    print(f"实验参数: {args}")

    set_seed(args.seed)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
    dataset = Planetoid(path, args.dataset, transform=T.ToSparseTensor())
    full_data = dataset[0].to(device)

    # --- 通用数据划分流程 ---
    train_indices = np.where(full_data.train_mask.cpu().numpy())[0]
    train_labels = full_data.y.cpu().numpy()[train_indices]
    client_private_indices = label_dirichlet_partition(train_labels, args.n_trainer, args.iid_beta)

    print("正在为每个客户端创建L-hop子图...")
    row, col, _ = full_data.adj_t.coo()
    global_edge_index = torch.stack([row, col], dim=0)
    communicate_indexes, in_com_train_indices, edge_indexes_clients = get_in_comm_indexes(
        global_edge_index.to('cpu'), client_private_indices, args.n_trainer,
        args.num_hops, torch.from_numpy(train_indices), num_nodes=full_data.num_nodes
    )
    print("子图创建完毕。")

    # --- 根据模型选择，执行不同的初始化 ---
    comm_info_for_eval = None
    if args.model == 'FedCSI':
        print("\n--- 初始化 Fed-CSI 模型及社区信息 ---")
        G = to_networkx(full_data, to_undirected=True)
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        print(f"在全局图上发现 {num_communities} 个社区。")
        node_to_comm_map = torch.tensor([partition.get(i, 0) for i in range(full_data.num_nodes)], dtype=torch.long).to(device)
        comm_graph = nx.Graph()
        comm_graph.add_nodes_from(range(num_communities))
        for u, v in G.edges():
            c_u, c_v = partition.get(u, -1), partition.get(v, -1)
            if c_u != -1 and c_v != -1 and c_u != c_v:
                if comm_graph.has_edge(c_u, c_v): comm_graph[c_u][c_v]['weight'] += 1
                else: comm_graph.add_edge(c_u, c_v, weight=1)
        comm_data = from_networkx(comm_graph)
        comm_adj = torch_sparse.SparseTensor(
            row=comm_data.edge_index[0], col=comm_data.edge_index[1],
            value=comm_data.weight.float() if comm_data.edge_index.numel() > 0 else torch.tensor([]),
            sparse_sizes=(num_communities, num_communities)
        ).to(device)

        global_model = CrossScaleGNN(
            nfeat=dataset.num_features, nhid=args.num_hidden, nclass=dataset.num_classes,
            dropout=args.dropout, num_layers=args.num_layers, num_communities=num_communities
        ).to(device)

        clients = []
        for i in range(args.n_trainer):
            client_global_node_ids = communicate_indexes[i]
            clients.append(FedCSIClient(
                i, global_model,
                local_features=full_data.x[client_global_node_ids].to(device),
                local_adj=edge_indexes_clients[i].to(device),
                local_labels=full_data.y[client_global_node_ids].to(device),
                local_train_indices=in_com_train_indices[i].to(device),
                node_to_comm_map=node_to_comm_map[client_global_node_ids].to(device),
                args=args
            ))
    elif args.model == 'FedGCN':
        print("\n--- 初始化 FedGCN 模型 ---")
        global_model = GCN(
            nfeat=dataset.num_features, nhid=args.num_hidden, nclass=dataset.num_classes,
            dropout=args.dropout, num_layers=args.num_layers
        ).to(device)
        clients = []
        for i in range(args.n_trainer):
            nodes_for_client = communicate_indexes[i]
            clients.append(FedGCNClient(
                i, global_model,
                local_features=full_data.x[nodes_for_client].to(device),
                local_adj=edge_indexes_clients[i].to(device),
                local_labels=full_data.y[nodes_for_client].to(device),
                local_train_indices=in_com_train_indices[i].to(device),
                args=args
            ))
    else:
        raise ValueError(f"未知模型: {args.model}. 请选择 'FedGCN' 或 'FedCSI'.")

    print("\n--- 客户端初始化完毕 ---")
    for i, client in enumerate(clients):
        print(f"  客户端 {i + 1} 子图包含 {len(client.features)} 个节点，其中 {len(client.train_idx)} 个用于本地训练。")

    # --- 联邦学习主循环 (已加入验证集逻辑) ---
    print(f"\n--- {args.model} 联邦学习开始 ---")
    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    for round_idx in range(args.global_rounds):
        client_model_states = []

        if args.model == 'FedCSI':
            all_reps, all_ids = [], []
            for i, client in enumerate(clients):
                local_reps = client.generate_initial_reps(global_model.state_dict())
                all_reps.append(local_reps)
                all_ids.append(communicate_indexes[i])

            all_reps_cat = torch.cat(all_reps, dim=0)
            all_ids_cat = torch.cat(all_ids, dim=0).to(device)

            initial_node_reps = scatter(
                src=all_reps_cat,
                index=all_ids_cat,
                dim=0,
                reduce="mean",
                dim_size=full_data.num_nodes
            )
            with torch.no_grad():
                comm_features = scatter(src=initial_node_reps, index=node_to_comm_map, dim=0, reduce="mean")
                global_comm_reps = global_model.run_comm_gnn(comm_features, comm_adj)

            for client in clients:
                updated_state = client.train(global_model.state_dict(), global_comm_reps, comm_adj)
                client_model_states.append(updated_state)

            new_global_state = server_aggregate(global_model.state_dict(), client_model_states)
            comm_info_for_eval = (global_comm_reps, comm_adj, node_to_comm_map)

        elif args.model == 'FedGCN':
            for client in clients:
                updated_state = client.train(global_model.state_dict())
                client_model_states.append(updated_state)

            new_global_state = server_aggregate(global_model.state_dict(), client_model_states)
            comm_info_for_eval = None

        global_model.load_state_dict(new_global_state)

        val_acc = evaluate_model(global_model, full_data, full_data.val_mask, comm_info=comm_info_for_eval)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = copy.deepcopy(global_model.state_dict())
            patience_counter = 0  # 重置计数器
            print(f"回合 {round_idx + 1:03d} | 当前验证准确率: {val_acc:.4f} | *** 新的最佳验证准确率 ***")
        else:
            patience_counter += 1  # 增加计数器
            print(
                f"回合 {round_idx + 1:03d} | 当前验证准确率: {val_acc:.4f} | 最佳验证准确率: {best_val_accuracy:.4f} | 早停计数: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\n早停触发：验证集准确率在 {args.patience} 个回合内未提升。")
            break  # 中断训练循环

    print(f"\n--- {args.model} 联邦学习结束 ---")

    # --- 最终测试：使用验证集上最好的模型 ---
    print("\n--- 加载在验证集上表现最好的模型进行最终测试 ---")
    if best_model_state:
        global_model.load_state_dict(best_model_state)

        # 最终测试需要重新计算一次最优模型对应的社区表示
        if args.model == 'FedCSI':
             with torch.no_grad():
                all_reps, all_ids = [], []
                for i, client in enumerate(clients):
                    local_reps = client.generate_initial_reps(global_model.state_dict())
                    all_reps.append(local_reps)
                    all_ids.append(communicate_indexes[i])
                all_reps_cat = torch.cat(all_reps, dim=0)
                all_ids_cat = torch.cat(all_ids, dim=0)
                initial_node_reps = torch.zeros(full_data.num_nodes, args.num_hidden).to(device)
                initial_node_reps[all_ids_cat] = all_reps_cat
                comm_features = scatter(src=initial_node_reps, index=node_to_comm_map, dim=0, reduce="mean")
                final_comm_reps = global_model.run_comm_gnn(comm_features, comm_adj)
                comm_info_for_eval = (final_comm_reps, comm_adj, node_to_comm_map)

        final_test_acc = evaluate_model(global_model, full_data, full_data.test_mask, comm_info=comm_info_for_eval)
        print(f"最终测试集准确率: {final_test_acc:.4f}")
    else:
        print("训练失败，未能产生最佳模型。")


#实验参数: Namespace(model='FedCSI', dataset='cora', global_rounds=200, local_step=5, learning_rate=0.0008, weight_decay=0.0005,
#cora    n_trainer=10, num_layers=2, num_hops=2, num_hidden=128, dropout=0.5, patience=20, gpu=True, iid_beta=10000, seed=42)