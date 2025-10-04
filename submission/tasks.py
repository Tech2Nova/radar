# submission/tasks.py
from celery import shared_task
from .models import Task
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from django.conf import settings  # 用于获取BASE_DIR

logger = logging.getLogger(__name__)

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ACTIONS = 2  # 正常/异常


# 定义 DQN_LSTM 模型（与您的代码一致）
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
        out, (h, c) = self.lstm(x)  # out: [B, seq_len, hidden_dim]
        pooled = out.mean(dim=1)  # [B, hidden_dim]
        out = self.net(pooled)
        return self.fc(out)


# 读取测试数据（修改为动态文件路径）
def load_test_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        sample = data[:, 1:17]  # 忽略 index 列，只取数值列（假设 16 个特征）
        if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
            raise ValueError("数据包含 NaN 或 Inf 值，无法处理。")
        return sample.astype(np.float32)
    except Exception as e:
        raise ValueError(f"读取 {file_path} 时出错: {e}")


@shared_task
def process_file_task(task_id, files):
    logger.info(f"Processing task {task_id} with files: {files}")
    try:
        task = Task.objects.get(task_id=task_id)

        # 假设只处理第一个文件（单个测试文件）
        if not files:
            raise ValueError("No files to process")
        file_name = files[0]  # 文件名，如 "uuid.csv"
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)  # 完整路径

        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")

        # 加载模型（假设 best_model.pth 在项目根目录）
        model_path = os.path.join(settings.BASE_DIR, 'best_model.pth')
        input_dim = 16  # 特征维度
        model = DQN_LSTM(input_dim).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # 加载数据并预测
        test_data = load_test_data(file_path)
        state_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = model(state_tensor)
            probs = F.softmax(q_values, dim=1)
            action = q_values.argmax(dim=1).item()  # 0 或 1
            confidence = probs.max(dim=1)[0].item()

        # 更新 Task
        task.result = action
        task.confidence = confidence
        task.status = "completed"
        task.save()
        logger.info(f"Task {task_id} completed with result: {action}, confidence: {confidence}")

    except Task.DoesNotExist:
        logger.error(f"Task {task_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        if task:
            task.status = "failed"
            task.save()
        raise