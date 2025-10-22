from typing import List
from torch.utils.data import Subset
from collections import defaultdict
import torch
import numpy as np
import random

def _get_local_targets_and_indices(dataset):
    """
    返回 (labels, local_indices)
    - 若 dataset 是原始数据集：labels = [dataset[i][1]], local_indices = range(len(dataset))
    - 若 dataset 是 Subset：labels 只取该 Subset 的条目，local_indices = range(len(subset))
    这样后续的所有划分都在“局部索引空间”进行，避免越界。
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        # 先尽量从 base 的属性里拿标签（快），拿不到就逐条取
        if hasattr(base, "targets"):
            base_targets = base.targets
            if torch.is_tensor(base_targets):
                base_targets = base_targets.tolist()
            labels = [base_targets[i] for i in dataset.indices]
        elif hasattr(base, "labels"):
            base_targets = base.labels
            if torch.is_tensor(base_targets):
                base_targets = base_targets.tolist()
            labels = [base_targets[i] for i in dataset.indices]
        else:
            labels = [dataset[i][1] for i in range(len(dataset))]
        local_indices = list(range(len(dataset)))
        return labels, local_indices
    else:
        # 原始数据集：直接在本地空间构造
        if hasattr(dataset, "targets"):
            t = dataset.targets
            labels = t.tolist() if torch.is_tensor(t) else list(t)
        elif hasattr(dataset, "labels"):
            t = dataset.labels
            labels = t.tolist() if torch.is_tensor(t) else list(t)
        else:
            labels = [dataset[i][1] for i in range(len(dataset))]
        local_indices = list(range(len(dataset)))
        return labels, local_indices

def _even_sizes(total: int, parts: int) -> List[int]:
    base = total // parts
    rem  = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]

def _assign_class_groups(num_classes: int, client_num: int) -> List[List[int]]:
    classes = list(range(num_classes))
    groups: List[List[int]] = [[] for _ in range(client_num)]
    if client_num <= num_classes:
        sizes = _even_sizes(num_classes, client_num)  # 每人拿多少类
        ptr = 0
        for i, k in enumerate(sizes):
            groups[i] = classes[ptr:ptr+k]
            ptr += k
    else:
        for i, c in enumerate(classes):
            groups[i % client_num].append(c)
    return groups

def fashion_mnist_preprocess(
    mnist,
    client_num: int,
    mode: str = "iid",      # "iid" | "noniid"
    sample_rate: float = 1.0,
    dominant_frac: float = 0.7,
    seed: int = 42,
    classes_per_client: int = None,  # 控制每个客户端拥有的类别数（仅在 noniid 使用）
):
    """
    当 mode="noniid" 且 classes_per_client 指定时：
      - 将每个类别的样本先切成若干“片”(shard)，每片只含单一类别；
      - 分配给客户端，使每个客户端恰好拥有 classes_per_client 个不同类别；
      - 客户间不重复样本，且恰好用尽 total_target = round(N * sample_rate) 个样本。
    其他 mode 行为与原逻辑一致。
    """
    rng = torch.Generator().manual_seed(seed)

    # === 取标签与局部索引空间 ===
    labels, local_all = _get_local_targets_and_indices(mnist)
    N = len(local_all)

    # 本轮要使用的目标样本总量（与原实现一致）
    total_target = int(round(N * sample_rate))
    per_client_sizes = _even_sizes(total_target, client_num)

    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    # 各类对应的（局部）样本索引池，先打乱
    idx_by_class = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in idx_by_class:
        perm = torch.randperm(len(idx_by_class[c]), generator=rng)
        idx_by_class[c] = [idx_by_class[c][p.item()] for p in perm]

    # 全局池（局部索引）打乱，用于后续补齐/裁剪
    global_pool = torch.tensor(local_all)
    global_pool = global_pool[torch.randperm(N, generator=rng)]

    client_indices: List[List[int]] = [[] for _ in range(client_num)]
    mode_l = mode.lower()

    if mode_l == "iid" or (mode_l == "noniid" and not classes_per_client):
        # === 原 IID 或未指定 classes_per_client 时，保持你原来的逻辑 ===
        if mode_l == "iid":
            ptr = 0
            for i, k in enumerate(per_client_sizes):
                client_indices[i] = global_pool[ptr:ptr+k].tolist()
                ptr += k
        else:
            # 沿用你原来的 noniid 分组逻辑
            groups = _assign_class_groups(num_classes, client_num)
            class_to_clients = defaultdict(list)
            for cid, cls_list in enumerate(groups):
                for c in cls_list:
                    class_to_clients[c].append(cid)

            for c, owners in class_to_clients.items():
                cidx = torch.tensor(idx_by_class[c], dtype=torch.long)
                sizes = _even_sizes(len(cidx), len(owners))
                ptr = 0
                for owner, s in zip(owners, sizes):
                    part = cidx[ptr:ptr+s].tolist()
                    client_indices[owner].extend(part)
                    ptr += s

            used = set(i for lst in client_indices for i in lst)
            remaining_pool = [x for x in global_pool.tolist() if x not in used]
            for i in range(client_num):
                need = per_client_sizes[i] - len(client_indices[i])
                if need < 0:
                    random.shuffle(client_indices[i])
                    client_indices[i] = client_indices[i][:per_client_sizes[i]]
                elif need > 0:
                    take = min(need, len(remaining_pool))
                    client_indices[i].extend(remaining_pool[:take])
                    remaining_pool = remaining_pool[take:]

    elif mode_l == "noniid" and classes_per_client:
        K = int(classes_per_client)
        assert K >= 1, "classes_per_client 必须为正整数"

        # === 计算“每个类别要被切成多少片” ===
        # 理想：每类切 K 片（满足你给的例子）；当 client_num != num_classes 时，
        # 为了让总片数 == client_num*K，我们把片数尽量均匀地分配给各类（K 或 K±1）。
        total_shards_needed = client_num * K
        shards_per_class_list = _even_sizes(total_shards_needed, num_classes)  # sum 等于 total_shards_needed

        # 若 client_num == num_classes，优先严格按“每类 K 片”
        if client_num == num_classes:
            shards_per_class_list = [K] * num_classes

        # === 把每个类别切成若干片 ===
        def _split_list(lst, parts: int) -> List[List[int]]:
            if parts <= 0:
                return []
            sizes = _even_sizes(len(lst), parts)
            out, p = [], 0
            for s in sizes:
                out.append(lst[p:p+s])
                p += s
            return out

        shards_by_class = {c: _split_list(idx_by_class[c], shards_per_class_list[c]) for c in range(num_classes)}

        # === 分配：保证每个客户端恰好 K 个不同类别 ===
        client_classes = [set() for _ in range(client_num)]
        client_buckets: List[List[int]] = [[] for _ in range(client_num)]
        remain_slots = [K] * client_num

        # 为了更稳的分配：循环各类，逐片分配到“当前类别未拥有且尚有空位”的客户端
        class_order = list(range(num_classes))
        random.shuffle(class_order)

        # 记录还有片未分完的类别集合
        remaining_classes = set(c for c in range(num_classes) if len(shards_by_class[c]) > 0)

        # 贪心分配直至所有片分完
        while remaining_classes:
            progress = False
            # 为了均衡，按“当前已分配类别数”少的客户端优先
            clients_order = sorted(range(client_num), key=lambda i: (len(client_classes[i]), i))

            for c in list(remaining_classes):
                shard_list = shards_by_class[c]
                if not shard_list:
                    remaining_classes.discard(c)
                    continue

                # 尝试把本类的一片分给一个合适的客户端
                assigned = False
                for i in clients_order:
                    if remain_slots[i] <= 0 or (c in client_classes[i]):
                        continue
                    # 分配这一片
                    part = shard_list.pop(0)
                    client_buckets[i].extend(part)
                    client_classes[i].add(c)
                    remain_slots[i] -= 1
                    assigned = True
                    progress = True
                    if not shard_list:
                        remaining_classes.discard(c)
                    break

                # 如果这一轮没法把该类的片分出去（大家都满或都已有该类），先跳过，等下一轮
            if not progress:
                # 若进入僵局（例如某些类剩余片数过多），放宽策略：优先给“未满但已有该类”的客户端，
                # 这样仍不引入新类别（保持每人最多 K 类），但把剩余样本灌入已有的类中。
                for c in list(remaining_classes):
                    shard_list = shards_by_class[c]
                    if not shard_list:
                        remaining_classes.discard(c)
                        continue
                    for i in range(client_num):
                        if remain_slots[i] <= 0:
                            continue
                        # 只允许把该类的剩余片灌到“已经拥有该类”的客户端，避免增加新类数
                        if c in client_classes[i]:
                            part = shard_list.pop(0)
                            client_buckets[i].extend(part)
                            # remain_slots[i] 不变（不增加类别数）
                            progress = True
                            if not shard_list:
                                remaining_classes.discard(c)
                            break
                if not progress:
                    # 极端情况下仍无法推进（几乎不会在 10 类/常见设置出现），这里直接打散给“未满”的客户端，
                    # 可能会让个别客户端多于 K 类（尽量避免，但保底保证“全部数据被使用”）。
                    for c in list(remaining_classes):
                        shard_list = shards_by_class[c]
                        while shard_list:
                            any_assigned = False
                            for i in range(client_num):
                                if remain_slots[i] > 0:
                                    part = shard_list.pop(0)
                                    client_buckets[i].extend(part)
                                    client_classes[i].add(c)
                                    remain_slots[i] -= 1
                                    any_assigned = True
                                    break
                            if not any_assigned:
                                # 所有人都满了，只能再分到已有的人（增加类别数），作为最后兜底
                                i = random.randrange(client_num)
                                part = shard_list.pop(0)
                                client_buckets[i].extend(part)
                        remaining_classes.discard(c)

        # === 到这里，client_buckets 已经装满样本（不重复），但数量可能与 per_client_sizes 不完全一致。
        # 我们做两步：“保类裁剪 + 同类补齐”，确保
        #   1) 每个客户端保留其 K 个类别（不把某个类别裁没），
        #   2) 精确满足 per_client_sizes，且不引入新类别。
        used_after_initial = set(x for lst in client_buckets for x in lst)
        # 建立剩余池（未被使用的索引）
        remaining_pool = [x for x in global_pool.tolist() if x not in used_after_initial]

        # 方便按类过滤
        def _idx_label(i_local):
            return int(labels[i_local])

        # 先“保类裁剪”：对超出的客户端，只从“某个类别中保留至少 1 个样本”的前提下裁剪
        for i in range(client_num):
            cur = client_buckets[i]
            target = per_client_sizes[i]
            if len(cur) <= target:
                continue
            # 按类别分桶
            by_c = defaultdict(list)
            for idx in cur:
                by_c[_idx_label(idx)].append(idx)
            # 每个已有类别先保留 1 个样本
            keep = []
            removable = []
            for c, lst in by_c.items():
                keep.append(lst[0])
                removable.extend(lst[1:])
            # 还需要保留的名额
            need_more = target - len(keep)
            if need_more < 0:
                # 极端：目标太小，随即裁剪 keep（尽量均匀）
                random.shuffle(keep)
                keep = keep[:target]
                removable = []
            else:
                random.shuffle(removable)
                keep.extend(removable[:need_more])
            client_buckets[i] = keep
            # 把被裁剪掉的放回剩余池
            removed = set(cur) - set(keep)
            remaining_pool.extend(list(removed))

        # 再“同类补齐”：不足的客户端，只从“他已拥有的类别”的未使用样本中补
        # 构建一个从类到未使用样本的查找表，便于过滤
        rem_by_class = defaultdict(list)
        for idx in remaining_pool:
            rem_by_class[_idx_label(idx)].append(idx)
        for c in rem_by_class:
            random.shuffle(rem_by_class[c])

        for i in range(client_num):
            cur = client_buckets[i]
            target = per_client_sizes[i]
            if len(cur) >= target:
                continue
            owned = set(_idx_label(x) for x in cur)
            need = target - len(cur)
            # 仅从已拥有的类别中补齐
            added = []
            for c in list(owned):
                if need <= 0:
                    break
                take = min(need, len(rem_by_class[c]))
                if take > 0:
                    added.extend(rem_by_class[c][:take])
                    rem_by_class[c] = rem_by_class[c][take:]
                    need -= take
            # 若仍不足（极少发生），从任意类里补（这会引入新类，作为兜底）
            if need > 0:
                # 把 rem_by_class 拉平成列表
                flat_rest = []
                for c, lst in rem_by_class.items():
                    flat_rest.extend(lst)
                random.shuffle(flat_rest)
                take = min(need, len(flat_rest))
                added.extend(flat_rest[:take])
                # 从 rem_by_class 中删除这些
                taken_set = set(added)
                for c in list(rem_by_class.keys()):
                    rem_by_class[c] = [x for x in rem_by_class[c] if x not in taken_set]

            client_buckets[i].extend(added)

        client_indices = [sorted(set(lst)) for lst in client_buckets]

    elif mode_l == "mixed_noniid":
        # === 保留你原来的 mixed_noniid 实现 ===
        groups = _assign_class_groups(num_classes, client_num)
        dominant_pools = []
        for cls_list in groups:
            pool = []
            for c in cls_list:
                pool.extend(idx_by_class[c])
            pool = torch.tensor(pool, dtype=torch.long)
            if len(pool) > 0:
                perm = torch.randperm(len(pool), generator=rng)
                pool = pool[perm]
            dominant_pools.append(pool)

        used = set()
        for i in range(client_num):
            target_size = per_client_sizes[i]
            dom_need = int(round(dominant_frac * target_size))
            taken = 0
            if len(dominant_pools[i]) > 0:
                for idx in dominant_pools[i].tolist():
                    if idx in used:
                        continue
                    client_indices[i].append(idx)
                    used.add(idx)
                    taken += 1
                    if taken >= dom_need:
                        break
        remaining_pool = [x for x in global_pool.tolist() if x not in used]
        ptr = 0
        for i in range(client_num):
            need = per_client_sizes[i] - len(client_indices[i])
            if need > 0:
                client_indices[i].extend(remaining_pool[ptr:ptr+need])
                ptr += need
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # === 把“局部索引”映射回 Subset ===
    subsets = []
    if isinstance(mnist, Subset):
        base_indices = mnist.indices
        for idxs in client_indices:
            idxs = sorted(set(idxs))
            mapped = [base_indices[i] for i in idxs]
            subsets.append(Subset(mnist.dataset, mapped))
    else:
        for idxs in client_indices:
            idxs = sorted(set(idxs))
            subsets.append(Subset(mnist, idxs))
    return subsets

