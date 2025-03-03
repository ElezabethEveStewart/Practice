import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import tqdm

# %% 读取数据并预处理
df = pd.read_csv("angles_by_hour.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 仅保留 'y' 作为输入
y_data = df['y'].values

# 定义时间窗口大小
seq_length = 60  # 你可以调整这个值


# 自定义 Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 创建 Dataset 和 DataLoader
dataset = TimeSeriesDataset(y_data, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


# %% 定义模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_batch):
        h0 = torch.zeros(1, x_batch.size(0), hidden_size).to(x_batch.device)
        out, _ = self.rnn(x_batch, h0)
        out = self.linear(out[:, -1, :])  # 只取最后时刻的输出
        return out


input_size = 1  # 如果每个时间步的特征只有一个
hidden_size = 20
output_size = 1
#
# # 实例化模型
# model = SimpleRNN(input_size, hidden_size, output_size)
#
# # %% 设置损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 100

# # 训练模型
# for epoch in tqdm.tqdm(range(num_epochs)):
#     model.train()
#
#     for x_batch, y_batch in dataloader:
#         x_batch = x_batch.unsqueeze(2)  # 添加一个维度，使其变为 (batch_size, seq_length, input_size)
#
#         optimizer.zero_grad()  # 清除之前的梯度
#         outputs = model(x_batch)  # 正向传播
#         loss = criterion(outputs, y_batch.unsqueeze(1))  # 计算损失，y_batch 需要调整为 (batch_size, 1)
#
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
#
#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch}, Loss: {loss.item()}")
#
# # %% 保存模型
# torch.save(model.state_dict(), "simple_rnn_model.pth")
# print("Model saved to simple_rnn_model.pth")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %% 加载模型
model = SimpleRNN(input_size=1, hidden_size=20, output_size=1)
model.load_state_dict(torch.load("simple_rnn_model.pth"))
model.eval()  # 设置为评估模式

# %% 进行预测并可视化
predictions = []
true_values = []

with torch.no_grad():  # 禁用梯度计算
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.unsqueeze(2)  # 重新调整输入形状
        outputs = model(x_batch)
        predictions.append(outputs.numpy())
        true_values.append(y_batch.numpy())

# 将预测值和真实值转换为 numpy 数组
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# 可视化预测值与真实值
plt.figure(figsize=(10, 6))
plt.plot(true_values, label="True Values", color='blue')
plt.plot(predictions, label="Predictions", color='red', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('True Values vs Predictions')
plt.legend()
plt.show()
