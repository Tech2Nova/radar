import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ACTIONS = 2  # 正常/异常

# 定义 DQN_LSTM 模型（与训练脚本一致）
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
        # mean pooling
        pooled = out.mean(dim=1)  # [B, hidden_dim]
        out = self.net(pooled)
        return self.fc(out)

# 读取测试数据（类似于 load_hpc_data，但针对单个文件）
def load_test_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        sample = data[:, 1:17]  # 忽略 index 列，只取数值列（假设 16 个特征）
        # 检查 NaN 或 Inf
        if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
            raise ValueError("数据包含 NaN 或 Inf 值，无法处理。")
        return sample.astype(np.float32)
    except Exception as e:
        raise ValueError(f"读取 {file_path} 时出错: {e}")

# 主函数
def main():
    # 加载模型
    input_dim = 16  # 特征维度，与训练一致
    model = DQN_LSTM(input_dim).to(DEVICE)
    model.load_state_dict(torch.load('best_model1004.pth', map_location=DEVICE))
    model.eval()  # 设置为评估模式

    # 读取测试文件
    test_file = "test.csv"
    if not os.path.exists(test_file):
        print(f"错误：文件 {test_file} 不存在！")
        return

    test_data = load_test_data(test_file)
    state_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, seq_len, input_dim]

    # 预测
    with torch.no_grad():
        q_values = model(state_tensor)  # [1, 2]
        probs = F.softmax(q_values, dim=1)  # softmax 得到概率
        action = q_values.argmax(dim=1).item()  # 0 或 1
        confidence = probs.max(dim=1)[0].item()  # 最大概率作为置信度

    # 输出结果
    label = "良性 (0)" if action == 0 else "恶意 (1)"
    print(f"预测结果：{action} ({label})")
    print(f"置信度：{confidence:.4f}")

if __name__ == "__main__":
    main()