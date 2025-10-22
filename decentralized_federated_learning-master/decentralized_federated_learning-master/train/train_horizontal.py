from concurrent.futures import ThreadPoolExecutor
from data.dataloader import DataLoader as DL
from data.preprocess import fashion_mnist_preprocess
from utlis.file_utls.yml_utils import read_yaml
from model.client.client_horizontal import Client
from model.model.resnet import ResNet, ResidualUnit
import time
import matplotlib.pyplot as plt
import os
import json
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import argparse
from datetime import datetime

# region algorithm description
"""
    Sparse Recovery Decentralized Federated Learning：
    1.获取Fashion-mnist数据集
    2.为每个客户端划分数据集，不同客户端拥有一定数量的数据集
    3.为客户端分配邻居节点，用于梯度数据的更新
    4.开始训练
        4.1 每个batch先进行正向传播与反向传播
        4.2 更新各自梯度信息
        4.3 梯度下降
"""
# endregion

# 定义多线程任务
def evaluate_global_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    model.train()
    return accuracy


if __name__ == '__main__':
    # region configuration
    parser = argparse.ArgumentParser(description="Run Decentralized Federated Learning Experiments.")
    parser.add_argument('--split_mode', type=str, choices=['iid', 'noniid'],)
    parser.add_argument('--batch_select', type=str)
    parser.add_argument('--client_num', type=int)
    parser.add_argument('--subset_frac', type=float)
    parser.add_argument('--classes_per_client', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--local_epochs', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--initial_neighbor_rate', type=float)
    parser.add_argument('--participation_rate', type=float)
    parser.add_argument('--communication_rounds', type=int, nargs='+')
    parser.add_argument('--s_ratio', type=float)
    parser.add_argument('--min_dense_params', type=float)
    parser.add_argument('--use_fedbn', type=str)
    parser.add_argument('--lr_decay_step', type=int)
    parser.add_argument('--lr_decay_beta', type=float)
    parser.add_argument('--dense_keywords', type=str)
    args = parser.parse_args()
    start_time = time.time()
    # 1.加载配置文件并获取配置信息
    config                = read_yaml('horizontal_fl')
    data_path             = config['data_path']
    split_mode            = args.split_mode if args.split_mode is not None else config['split_mode']
    batch_select          = args.batch_select if args.batch_select is not None else config['batch_select']
    client_num            = args.client_num if args.client_num is not None else config['client_num']
    subset_frac           = args.subset_frac if args.subset_frac is not None else config['subset_frac']
    classes_per_client    = args.classes_per_client if args.classes_per_client is not None else config['classes_per_client']
    lr                    = args.lr if args.lr is not None else config['lr']
    epochs                = args.epochs if args.epochs is not None else config['epochs']
    local_epochs          = args.local_epochs if args.local_epochs is not None else config['local_epochs']
    bs                    = args.bs if args.bs is not None else config['bs']
    gamma                 = args.gamma if args.gamma is not None else config['gamma']
    initial_neighbor_rate = args.initial_neighbor_rate if args.initial_neighbor_rate is not None else config['initial_neighbor_rate']
    participation_rate    = args.participation_rate if args.participation_rate is not None else config['participation_rate']
    communication_rounds  = args.communication_rounds if args.communication_rounds is not None else config['communication_rounds']
    s_ratio               = args.s_ratio if args.s_ratio is not None else config['s_ratio'] #transmission rate
    min_dense_params      = args.min_dense_params if args.min_dense_params is not None else config['min_dense_params']
    use_fedbn             = args.use_fedbn if args.use_fedbn is not None else config['use_fedbn']
    lr_decay_step         = args.lr_decay_step if args.lr_decay_step is not None else config['lr_decay_step']
    lr_decay_beta         = args.lr_decay_beta if args.lr_decay_beta is not None else config['lr_decay_beta']
    dense_keywords        = args.dense_keywords if args.dense_keywords is not None else config['dense_keywords']
    # endregion

    if len(communication_rounds) != client_num:
        raise ValueError("The length of communication_rounds and client_num should be equal!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dl_test = DL(data_path)
    # 尺寸要与模型输入匹配，这里假设是 [32, 32]
    global_test_set = dl_test.load_horizontal_data(train=False, resize=[32, 32])
    test_batch_size = config.get('test_batch_size', 128)
    global_test_loader = DataLoader(global_test_set, batch_size=test_batch_size, shuffle=False)
    # 2.加载训练数据
    dl = DL(data_path)
    mnist = dl.load_horizontal_data(True, [32, 32])

    torch.manual_seed(42)  # 为了可复现
    k = int(len(mnist) * subset_frac)  # 20%
    idx = torch.randperm(len(mnist))[:k].tolist()
    mnist = Subset(mnist, idx)  # 之后再去切客户端
    # 3.数据处理
    if split_mode == "iid":
        clients_data = fashion_mnist_preprocess(mnist, client_num, mode="iid")
    elif split_mode == "noniid":
        clients_data = fashion_mnist_preprocess(mnist, client_num, mode="noniid", classes_per_client=classes_per_client)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # 4.创建client_num个client
    """
    Client(client_data,net,idx)
        client_data：客户端数据集
        net：训练的网络，这里使用 resnet18 ==> Resnet(block,layers,num_classes,lr,epochs,bs,gamma)
            block：使用瓶颈结构BottleNeck还是残差单元ResidualUnit
            layers：当前结构每层有多少个block
            num_classes：分类数
            lr：学习率
            epochs：训练次数
            bs：batch_size
            gamma：SGD参数
        idx：当前client的编号
    """
    """
        若想使用resnet其他层数结构，若为深层resnet：
        1.将第一个参数残差块ResidualUnit改为瓶颈结构BottleNeck
        2.根据论文填写第二个参数层次[layer1,layer2...]
    """
    # 创建每个Client时：


    # 4.创建 client_num 个 client（每个 client 拥有独立 net）
    clients = []
    for i in range(client_num):
        net = ResNet(ResidualUnit, [2, 2, 2, 2], 10, lr, epochs, local_epochs, bs, gamma)
        # 将配置下发到模型上，Client.__init__ 会读取这些属性
        net.s_ratio = max(0.0, min(1.0, s_ratio))  # clamp 到 [0,1]
        net.min_dense_params = min_dense_params
        net.dense_keywords = dense_keywords
        clients.append(Client(clients_data[i], net, i, communication_rounds[i], participation_rate, use_fedbn=use_fedbn, batch_select=batch_select))

    global_model = ResNet(ResidualUnit, [2, 2, 2, 2], 10, lr, epochs, local_epochs, bs, gamma).to(device)
    # 5.为每个client分配邻居节点，这里采用SRDFL文章中的方法：每个节点先根据initial_neighbor_rate建立一个邻居集合
    for i, client_i in enumerate(clients):
        # 内层循环遍历所有可能的邻居节点
        for j, client_j in enumerate(clients):
            # 确保不会把自己加为邻居
            if i == j:
                continue
            # 以 initial_neighbor_rate 的概率决定是否将 client_j 添加为 client_i 的邻居
            if random.random() < initial_neighbor_rate:
                client_i.neighbors.append(client_j)
    # 打印每个客户端的邻居数量，方便检查
    for client in clients:
        print(f"client {client.idx} initial number of neighbors: {len(client.neighbors)}")
    print("\n training begins! (epoch-by-epoch parallel mode with global evaluation)")

    global_accuracy_history = []
    local_accuracy_histories = {client_id: [] for client_id in range(client_num)}
        # 主线程的“每日”循环
    for epoch in range(epochs):
            print(f"\n--- global epoch {epoch + 1}/{epochs} ---")

            if (epoch + 1) % lr_decay_step == 0 and epoch > 0:
                new_lr = -1  # 用于打印
                # 遍历所有客户端，更新他们优化器中的学习率
                for client in clients:
                    for param_group in client.opt.param_groups:
                        param_group['lr'] *= lr_decay_beta
                        new_lr = param_group['lr']  # 获取更新后的新学习率
                print(f"--- At epoch {epoch + 1}, learning rate decays to {new_lr:.6f} ---")
            # 6.1 派发当轮的并行任务
            with ThreadPoolExecutor(max_workers=client_num) as executor:
                # 确保 client.py 中有 train_one_round 方法
                tasks = [executor.submit(client.train_one_round, epoch) for client in clients]
                # 6.2 等待并收集每个客户端返回的本地准确率
                local_accuracies_this_round = [task.result() for task in tasks]
                for client_id, local_acc in enumerate(local_accuracies_this_round):
                    local_accuracy_histories[client_id].append(local_acc)
            # 6.3 所有客户端完成一轮工作，主线程开始全局评估
            # 聚合所有客户端的模型权重到全局模型
            from collections import OrderedDict
            def _is_bn(name: str):
                n = name.lower()
                return ('bn' in n) or ('running_mean' in n) or ('running_var' in n) or ('num_batches_tracked' in n)

            with torch.no_grad():
                global_sd = OrderedDict()
                template_sd = clients[0].net.state_dict()
                for name, param in template_sd.items():
                    if _is_bn(name):
                        # FedBN: 全局评估模型不聚合BN，保留模板端（或任选一端）
                        global_sd[name] = template_sd[name].clone()
                        continue
                    if param.is_floating_point():
                        avg = None
                        for c in clients:
                            p = c.net.state_dict()[name]
                            avg = p.clone() if avg is None else (avg + p)
                        avg.div_(len(clients))
                        global_sd[name] = avg
                    else:
                        global_sd[name] = template_sd[name].clone()
                global_model.load_state_dict(global_sd)

            # 在测试集上评估全局模型
            global_acc = evaluate_global_model(global_model, global_test_loader, device)
            global_accuracy_history.append(global_acc)
            # 6.4 打印统一、清晰的评估结果
            print(f"--- global epoch {epoch + 1}: global acc = {global_acc:.2f}% ---")
            local_acc_str = ", ".join([f"C{i}:{acc:.2f}%" for i, acc in enumerate(local_accuracies_this_round)])
            print(f"--- local epoch {epoch + 1}: local acc = [{local_acc_str}] ---")

        # 7. 训练结束，保存结果
    # in train_horizontal.py, at the end of the script

    # ... (在 end_time = time.time() 之后) ...
    end_time = time.time()
    elapsed = end_time - start_time
    h = int(elapsed // 3600);
    m = int((elapsed % 3600) // 60);
    s = int(elapsed % 60)
    print(f"Total runtime: {elapsed:.2f}s ({h}:{m:02d}:{s:02d})")

    print("\n saving results")
    output_dir = 'res'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 新增：生成唯一的文件名 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"clients_{client_num}-epochs_{local_epochs}-{timestamp}.json"

    results_data = {
        "global_accuracy_history": global_accuracy_history,
        "local_accuracy_histories": local_accuracy_histories,
        "config": config,
        # 也将本次运行的实际参数存入结果，方便追溯
        "elapsed_seconds": elapsed,  # << 新增
        "elapsed_hms": f"{h}:{m:02d}:{s:02d}",
        "run_params": {
            "client_num": client_num,
            "local_epochs": local_epochs
        }
    }

    # --- 修改：使用新的文件名 ---
    file_path = os.path.join(output_dir, filename)  # <-- 修改这行
    with open(file_path, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"results save in {os.path.abspath(file_path)}")