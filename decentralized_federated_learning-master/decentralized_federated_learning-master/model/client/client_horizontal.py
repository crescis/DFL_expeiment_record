import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model.model.resnet import ResNet
import random
import time
import threading


# 定义Client类
class Client(object):
    """
    1.client_data: tensor 客户端数据
    2.net: nn.Module 模型
    3.idx: 当前client的id
    """

    def __init__(self, client_data, net: nn.Module, idx, communication_rounds, participation_rate, use_fedbn,
                 batch_select):
        # 配置gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 基础参数
        self.data = client_data  # client训练
        self.net = net.to(self.device)  # 模型
        self.idx = idx  # client编号
        # 邻居
        self.neighbors = []  # 邻居列表
        self.neighbors_num = 0  # 邻居节点数
        # 迭代训练
        self.iter = 1  # 记录迭代次数，保证各client间同步
        self.weights = {}  # 存放接收到的权重
        self.weights_lock = threading.Lock()
        self.list_my_weights = []  # client自己模型的权重值列表
        self.communication_rounds = communication_rounds  # 保存通信周期
        self.participation_rate = participation_rate  # client每一轮通信选择的邻居比例
        self.opt = optim.SGD(self.net.parameters(), lr=self.net.lr, momentum=self.net.gamma)
        self.criterion = nn.CrossEntropyLoss()
        # ===== 稀疏传输配置 =====
        self.s_ratio = getattr(self.net, "s_ratio")  # Top-s 比例，默认10%，可在模型上覆写
        # 不稀疏（密集）的层关键词：BN统计/参数、首层conv、最后fc；你可按需增删
        self.dense_keywords = getattr(self.net, "dense_keywords")
        # 对“很小”的张量直接走密集（避免Top-k开销 & 精度抖动）
        self.min_dense_params = getattr(self.net, "min_dense_params")  # numel < 这个阈值则不稀疏
        self.use_fedbn = use_fedbn
        self.batch_select = batch_select.lower()
        self._p = 0  # periodic 指针，跨轮次累积
        self._periodic_batches = None  # 缓存固定顺序的小批次索引

    """
    get_batch_data：获取batch_data
    """

    def _is_bn(self, name: str) -> bool:
        n = name.lower()
        return ('bn' in n) or ('running_mean' in n) or ('running_var' in n) or ('num_batches_tracked' in n)

    def _should_keep_dense(self, name: str, tensor: torch.Tensor) -> bool:
        # 命中关键词、BN、或参数量太小 → 不稀疏（密集）
        if self._is_bn(name):
            return True
        if any(kw in name.lower() for kw in self.dense_keywords):
            return True
        if tensor.numel() < self.min_dense_params:
            return True
        return False

    def _pack_sparse_state(self, state_dict: dict):
        """
        将本端state_dict打包成“稀疏消息”：
          payload[name] = ("dense", cpu_tensor) 或 ("sparse", shape, cpu_idx_int64, cpu_vals)
        非浮点buffer始终走dense。
        """
        payload = {}
        with torch.no_grad():
            for name, p in state_dict.items():
                # 非浮点（buffer等）直接 dense
                if not p.is_floating_point():
                    payload[name] = ("dense", p.detach().clone().cpu())
                    continue

                # 需要密集保留的层 或 s_ratio>=1.0
                if self._should_keep_dense(name, p) or self.s_ratio >= 1.0:
                    payload[name] = ("dense", p.detach().clone().cpu())
                    continue

                # Top-s 稀疏
                flat = p.detach().flatten()
                k = max(1, int(self.s_ratio * flat.numel()))
                # 取绝对值 Top-k 的索引，再带符号取值
                topk_vals, topk_idx = torch.topk(flat.abs(), k, largest=True, sorted=False)
                chosen_vals = flat[topk_idx]
                payload[name] = ("sparse",
                                 tuple(p.shape),
                                 topk_idx.detach().to(torch.long).cpu(),
                                 chosen_vals.detach().cpu())
        return payload

    def _apply_sparse_on_base(self, base_tensor: torch.Tensor, item):
        """
        把一条 payload 应用到 base_tensor 上：
          - dense: 直接替换为对方的张量
          - sparse: 复制base，在稀疏索引处用对方值覆盖
        返回新tensor（保持device/dtype/shape）
        """
        kind = item[0]
        if kind == "dense":
            dense_tensor = item[1].to(device=base_tensor.device, dtype=base_tensor.dtype)
            return dense_tensor
        elif kind == "sparse":
            shape, idx_cpu, vals_cpu = item[1], item[2], item[3]
            out = base_tensor.detach().clone()
            if out.numel() == 0:
                return out
            flat = out.view(-1)
            idx = idx_cpu.to(flat.device, dtype=torch.long)
            vals = vals_cpu.to(flat.device, dtype=flat.dtype)
            flat[idx] = vals  # 只覆盖Top-s位置，其他位置保持本地
            return out.view(shape)
        else:
            raise ValueError("unknown payload kind")

    def _build_periodic_plan(self):
        """将本地数据索引按固定顺序切成 batch 列表，仅构建一次。"""
        if self._periodic_batches is not None:
            return
        n = len(self.data)
        bs = self.net.bs
        # 固定顺序：0..n-1；若你想“只打乱一次再固定”，可改成随机排列再切
        order = list(range(n))
        self._periodic_batches = [order[i:i + bs] for i in range(0, n, bs)]
        if len(self._periodic_batches) == 0:
            self._periodic_batches = [[]]

    def _fetch_batch_by_indices(self, idx_list):
        """根据索引列表取出一个 batch 的 (x, y) 张量并放到 device。"""
        xs, ys = [], []
        for j in idx_list:
            x, y = self.data[j]
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs, dim=0).to(self.device)
        y = torch.tensor(ys, dtype=torch.long).to(self.device)
        return x, y

    def get_batch_data(self):
        # 保留原有随机策略
        if self.batch_select == 'random':
            return DataLoader(self.data, self.net.bs, shuffle=True, num_workers=1, pin_memory=True, prefetch_factor=16, persistent_workers=True, drop_last=False)
        else:
            # periodic 模式下不使用 PyTorch DataLoader 返回整个 epoch，
            # 在 train_one_round 里按“指针”逐批取
            return None

    """

    """

    # 模型训练
    def train_one_round(self, epoch):
        # 先与邻居对齐轮次（所有人都完成上一轮）
        self.wait_iter(self.neighbors)

        # === 先聚合（若到通信轮次）===
        if epoch % self.communication_rounds == 0:
            print(f"--- client {self.idx} in epoch {epoch + 1} [aggregation then communication] ---")
            communication_subset = [n for n in self.neighbors if random.random() < self.participation_rate]
            if len(communication_subset) > 0:
                # 发送上一轮结束时的权重快照
                self.send_weights_to_neighbor(communication_subset)
                # 给其他线程发送/接收的机会；如果你实现了更稳的屏障，可以去掉这行
                # self.wait_weights(len(communication_subset))
                time.sleep(0.05)
                # 用收到的任何权重进行聚合（未收到也可空过）
                self.update_weights()

        # === 再训练（以聚合后的模型为起点）===
        batch_data = self.get_batch_data()
        self.net.train()
        correct, samples = 0, 0
        if self.batch_select == 'random':
            batch_data = self.get_batch_data()  # shuffle=True
            for _ in range(self.net.local_epochs):
                for x, y in batch_data:
                    y, x = y.view(x.shape[0]).to(self.device), x.to(self.device)
                    out = self.net(x)
                    loss = self.criterion(out, y)
                    loss.backward()
                    self.opt.step();
                    self.opt.zero_grad()
                    pred = torch.max(out, dim=1)[1]
                    correct += torch.sum(pred == y)
                    samples += x.shape[0]
        else:
            # === periodic selection ===
            self._build_periodic_plan()  # 只构建一次固定的小批次列表
            n_batches = len(self._periodic_batches)

            # 极端保护：本客户端没有样本
            if n_batches == 0:
                return 0.0

            # 与随机模式对齐：每个本地 epoch 都要遍历完全部 mini-batches
            for _ in range(self.net.local_epochs):
                for _ in range(n_batches):
                    idxs = self._periodic_batches[self._p]
                    self._p = (self._p + 1) % n_batches  # 指针环状前进；不在轮末重置，跨轮保持
                    if not idxs:  # 空批次保护（理论上不会发生）
                        continue

                    # 取出该批次，前向、反向、更新
                    x, y = self._fetch_batch_by_indices(idxs)
                    out = self.net(x)
                    loss = self.criterion(out, y)
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()

                    # 统计本地 acc
                    pred = torch.max(out, dim=1)[1]
                    correct += (pred == y).sum().item()
                    samples += x.size(0)

        # 本轮结束，推进迭代计数
        self.iter += 1
        return 0.0 if samples == 0 else float(100.0 * correct / samples)

    # 将梯度发送给邻居client
    def send_weights_to_neighbor(self, neighbors):
        with torch.no_grad():
            full_sd = self.net.state_dict()
            msg = self._pack_sparse_state(full_sd)  # ← 关键：打包成稀疏payload
        for neighbor in neighbors:
            neighbor.receive_weights_from_neighbor(self.idx, msg)

    def receive_weights_from_neighbor(self, idx, received_state_dict):
        # received_state_dict 是 dict[name] -> ("dense",t) 或 ("sparse",shape,idx,vals)
        with self.weights_lock:
            # 这里不做clone/to(device)，在重建时再搬运，避免多余开销
            self.weights[idx] = received_state_dict

    # 等待接受梯度
    def wait_weights(self, expected_num: int):  # 接收一个参数：期望收到的权重数量
        while True:
            if len(self.weights) != expected_num:  # 和期望数量比较
                continue
            else:
                break

    # 等待同一个iter
    def wait_iter(self, neighbor_list):
        while True:
            is_ready = True
            # 遍历传入的邻居列表
            for neighbor in neighbor_list:
                # 使用 >= 来避免“赛跑”问题
                # 检查邻居的迭代次数是否已经追上或超过了自己当前的迭代次数
                if not (neighbor.iter >= self.iter):
                    is_ready = False
                    break  # 一旦发现有邻居没准备好，就立刻退出内层循环
            if is_ready:
                return  # 所有邻居都准备好了，退出等待
            else:
                # 短暂休息一下，避免CPU空转，给其他线程执行的机会
                time.sleep(0.01)
                continue  # 继续下一轮检查

    # 更新梯度
    def update_weights(self):
        # 1) 原子快照并清空
        with self.weights_lock:
            weights_snapshot = self.weights
            self.weights = {}
        if len(weights_snapshot) == 0:
            return

        import copy
        from collections import OrderedDict
        with torch.no_grad():
            my_sd = self.net.state_dict()

            # 2) 用本地参数为底板，按对方payload覆写Top-s位置，重建每个邻居的完整权重
            neighbor_full_list = []
            for nb_id, payload in weights_snapshot.items():
                rebuilt = OrderedDict()
                for name, base_t in my_sd.items():
                    if name in payload:
                        rebuilt[name] = self._apply_sparse_on_base(base_t, payload[name])
                    else:
                        # 对方没传这个键 → 保留本地
                        rebuilt[name] = base_t.detach().clone()
                neighbor_full_list.append(rebuilt)

            # 3) 逐层聚合（简单等权平均；若需要可改成样本数加权）
            avg_sd = OrderedDict()
            for name, t in my_sd.items():
                # 可选：FedBN——BN不聚合，保留本地（强烈建议在non-IID时开启）
                # FedBN 开关：true 时 BN 不聚合；false 时按普通层聚合
                if self.use_fedbn and self._is_bn(name):
                    avg_sd[name] = t.detach().clone()
                    continue

                if t.is_floating_point():
                    acc = t.detach().clone()
                    for nb in neighbor_full_list:
                        acc.add_(nb[name])
                    acc.div_(1 + len(neighbor_full_list))
                    avg_sd[name] = acc
                else:
                    # 非浮点buffer保持本地（例如num_batches_tracked）
                    avg_sd[name] = t.detach().clone()

            self.net.load_state_dict(avg_sd)

