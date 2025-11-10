import os
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch

def set_global_seed(seed=42):
    random.seed(seed)                        # Python 随机种子
    np.random.seed(seed)                      # NumPy 随机种子
    torch.manual_seed(seed)                   # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)              # PyTorch GPU 随机种子
    torch.cuda.manual_seed_all(seed)          # 多GPU
    torch.backends.cudnn.deterministic = True # 让cuDNN确定性
    torch.backends.cudnn.benchmark = False   # 禁用优化算法选择随机性

# 在 main 前调用
set_global_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ACTIONS = 2  # 正常/异常
NUM_EPISODES = 20
LR = 1e-3
EPS_START, EPS_END = 1.0, 0.01
EPS_DECAY = 200
GAMMA = 0.8
BUFFER_CAPACITY = 1000
BATCH_SIZE = 16

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 位置编码（适配老版本 PyTorch，无 batch_first 依赖）----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)                     # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)           # 偶数
        pe[:, 1::2] = torch.cos(position * div_term)           # 奇数
        pe = pe.unsqueeze(1)                                   # [max_len, 1, d_model]
        self.register_buffer('pe', pe)                         # 不参与训练

    def forward(self, x_seq_first: torch.Tensor):
        """
        x_seq_first: [seq_len, batch, d_model]
        """
        seq_len = x_seq_first.size(0)
        x_seq_first = x_seq_first + self.pe[:seq_len]
        return self.dropout(x_seq_first)

# ---- 注意力池化（让模型学习“哪些时间步更重要”）----
class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x_bt: torch.Tensor):
        """
        x_bt: [batch, seq_len, d_model]
        return: [batch, d_model]
        """
        # [B,T,1] -> [B,T]
        attn_logits = self.score(x_bt).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)           # over time
        # 加权求和: [B,T,1] * [B,T,D] -> [B,D]
        pooled = torch.bmm(attn_weights.unsqueeze(1), x_bt).squeeze(1)
        return pooled

# ---- DQN-Transformer 主体（可直接替换原 DQN_LSTM）----
class DQN_Transformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_actions: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)              # 输入投影到 d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model, dropout)
        self.pool = AttentionPooling(d_model)                  # 也可换成 nn.AdaptiveAvgPool1d
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions)
        )

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        return: Q(s, ·) -> [batch, num_actions]
        """
        # 投影到 d_model
        x = self.proj(x)                                       # [B,T,D]
        # 变换为 [T,B,D] 以适配老版本 Transformer
        x = x.transpose(0, 1)                                  # [T,B,D]
        # 加位置编码 -> 编码
        x = self.posenc(x)                                     # [T,B,D]
        x = self.encoder(x)                                    # [T,B,D]
        # 回到 [B,T,D]
        x = x.transpose(0, 1)                                  # [B,T,D]
        # 注意力池化到 [B,D]
        x = self.pool(x)                                       # [B,D]
        # 动作价值
        q = self.head(x)                                       # [B,A]
        return q


# ----------------------------
# 网络定义
# ----------------------------
class DQN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc = nn.Linear(32, NUM_ACTIONS)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, (h, c) = self.lstm(x)  # out: [B, seq_len, hidden_dim]

        # ------ 池化操作 ------
        # mean pooling
        pooled = out.mean(dim=1)  # [B, hidden_dim]
        # max pooling (可替代)
        # pooled, _ = out.max(dim=1)

        # 全连接映射到动作空间
        out = self.net(pooled)
        return self.fc(out)

# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)   # 这里加上


# ----------------------------
# ε-greedy
# ----------------------------
# ----------------------------
# ε 策略
# ----------------------------
def epsilon_by_frame(frame_idx):
    """线性衰减 ε"""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)

def select_action(policy_net, state_tensor, frame_idx, strategy="linear"):
    if strategy == "linear":
        eps = epsilon_by_frame(frame_idx)
        if random.random() < eps:
            return random.randrange(NUM_ACTIONS)
        else:
            with torch.no_grad():
                return policy_net(state_tensor).argmax(dim=1).item()

    elif strategy == "boltzmann":
        # 温度随训练逐渐下降
        tau = max(0.1, 1.0 - frame_idx / 10000)
        with torch.no_grad():
            q_values = policy_net(state_tensor).cpu().numpy().flatten()
        exp_q = np.exp(q_values / tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(NUM_ACTIONS, p=probs)

# ----------------------------
# DQN训练一步
# ----------------------------
def train_dqn(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """
    用普通 ReplayBuffer 训练 DQN
    :param policy_net: 当前策略网络
    :param target_net: 目标网络
    :param replay_buffer: 普通经验回放池（返回 state, action, reward, next_state, done）
    :param optimizer: 优化器
    :param batch_size: 批大小
    :param gamma: 折扣因子
    :param device: 设备 ('cpu' 或 'cuda')
    """

    if len(replay_buffer) < batch_size:
        return None  # 缓存不够时不训练

    # 从 ReplayBuffer 采样
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # 转 tensor
    state      = torch.FloatTensor(state).to(device)
    action     = torch.LongTensor(action).unsqueeze(1).to(device)  # [B,1]
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # [B,1]
    next_state = torch.FloatTensor(next_state).to(device)
    done       = torch.FloatTensor(done).unsqueeze(1).to(device)    # [B,1]

    # 计算 Q(s,a)
    q_values = policy_net(state).gather(1, action)  # 取出执行过的动作的 Q 值

    # 计算 Q_target
    with torch.no_grad():
        max_next_q = target_net(next_state).max(1, keepdim=True)[0]  # 最大 Q 值
        expected_q = reward + (1 - done) * gamma * max_next_q

    # 计算损失 (MSELoss)
    criterion = nn.MSELoss()
    loss = criterion(q_values, expected_q)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ----------------------------
# 数据读取函数
# ----------------------------
def load_hpc_data(folder_path):
    X, y = [], []
    for label, cls in enumerate(["benign/", "malware/"]):
        cls_folder = os.path.join(folder_path, cls)
        for f in os.listdir(cls_folder):
            if f.endswith(".csv"):
                df = pandas.read_csv(os.path.join(cls_folder, f),header=0)
                # 只保留数值列，去掉 index 列
                numeric_df = df.iloc[:, 1:17]
                X.append(numeric_df.values.astype(np.float32))
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

def load_hpc_data_clean(folder_path):
    """
    读取 HPC CSV 数据，并删除包含 NaN/Inf 的样本
    """
    X, y = [], []
    for label, cls in enumerate(["benign/", "malware/"]):
        cls_folder = os.path.join(folder_path, cls)
        for f in os.listdir(cls_folder):
            if f.endswith(".csv"):
                try:
                    data = np.loadtxt(os.path.join(cls_folder, f), delimiter=",", skiprows=1)
                    sample = data[:, 1:17]  # 忽略 index 列
                    # 检查 NaN 或 Inf
                    if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
                        print(f"[!] Skipping sample {f} in class {cls} due to NaN/Inf")
                        continue
                    X.append(sample.astype(np.float32))
                    y.append(label)
                except Exception as e:
                    print(f"[!] Error reading {f}: {e}, skipping.")
                    continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

# ----------------------------
# 测试函数
# ----------------------------
def evaluate(policy_net, X, y):
    policy_net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(X)):
            state = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = policy_net(state).argmax(dim=1).item()
            y_true.append(y[i])
            y_pred.append(action)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Test - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        #self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Dropout(0.1),
                                 nn.Linear(100 * hidden_dim, 64),
                                 # nn.Tanh(),
                                 # nn.Dropout(0.1),
                                 nn.Linear(64, 32),
                                 nn.Linear(32, 2)
                                 )
        self.fc = nn.Linear(hidden_dim, NUM_ACTIONS)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, (h, c) = self.lstm(x)
        out = self.net(out)
        # out = out[:, -1, :]  # 取最后时间步输出
        # return self.fc(out)
        return out


def stratified_split(X, y, test_size=0.2, random_seed=42):
    """
    同时保证类别比例和划分可复现
    """
    np.random.seed(random_seed)
    X = np.array(X)
    y = np.array(y)

    train_idx, test_idx = [], []

    classes = np.unique(y)
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)  # 类内随机打乱
        split = int(len(cls_idx) * (1 - test_size))
        train_idx.extend(cls_idx[:split])
        test_idx.extend(cls_idx[split:])

    # 按索引顺序整理数据（可选）
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    X, y = load_hpc_data_clean("./collect_data/1ms")
    train_X, test_X, train_y, test_y = stratified_split(X, y, test_size=0.2, random_seed=42)

    # 转为Tensor
    train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
    test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LSTMClassifier(input_dim=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            if torch.isnan(loss) or torch.isinf(loss):
                print("[!] Warning: loss is NaN/Inf, skipping batch")
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(train_loader.dataset)

        # --------------------------
        # 每轮在训练集上评估指标
        # --------------------------
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = outputs.argmax(dim=1)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} | "
              f"Train Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # --------------------------
    # 最终在测试集上评估
    # --------------------------
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Test Results - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    main()

    X, y = load_hpc_data_clean("./collect_data/1ms")
    train_X, test_X, train_y, test_y = stratified_split(X, y, test_size=0.2, random_seed=42)

    # 确认输入维度

    input_dim = train_X.shape[2]
    print(f"Input dimension: {input_dim}")

    policy_net = DQN_LSTM(input_dim).to(DEVICE)
    target_net = DQN_LSTM(input_dim).to(DEVICE)


    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer()
    frame_idx = 0

    # DQN训练循环
    strategy = "boltzmann"  # 可选 r"linea" 或 "boltzmann"
    for episode in range(NUM_EPISODES):
        y_true_train, y_pred_train = [], []
        for i in range(len(train_X)):
            state = train_X[i]
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # 动态探索策略
            action = select_action(policy_net, state_tensor, frame_idx, strategy=strategy)
            frame_idx += 1

            reward = 1 if action == train_y[i] else -1
            done = True
            replay_buffer.push(state, action, reward, state, done)
            train_dqn(policy_net, target_net, optimizer, replay_buffer,batch_size=BATCH_SIZE,gamma=GAMMA,device=DEVICE)

            y_true_train.append(train_y[i])
            y_pred_train.append(action)

        # 每轮输出训练指标
        acc = accuracy_score(y_true_train, y_pred_train)
        prec = precision_score(y_true_train, y_pred_train)
        rec = recall_score(y_true_train, y_pred_train)
        f1 = f1_score(y_true_train, y_pred_train)
        print(f"Episode {episode} - Train Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    torch.save(policy_net.state_dict(), 'best_model1004.pth')
    print("saved model")

    # 测试评估
    evaluate(policy_net, test_X, test_y)
