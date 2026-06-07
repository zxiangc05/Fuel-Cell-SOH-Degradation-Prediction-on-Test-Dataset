import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('FC_Degradation_FTP75.csv', header=None)
data.columns = ['Current', 'Voltage', 'Power', 'SOH']

# 2. 数据归一化 (神经网络最喜欢 0-1 之间的数据)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 选取 电流、电压、功率 作为输入特征 (X)，SOH 作为预测目标 (y)
X = scaler_X.fit_transform(data[['Current', 'Voltage', 'Power','SOH']].values)
y = scaler_y.fit_transform(data[['SOH']].values)

# 3. 构造时间序列滑动窗口 (极其关键！)
# 比如：用过去 50 秒的数据，预测下一秒的 SOH
seq_length = 50 
X_seq, y_seq = [], []
for i in range(len(X) - seq_length):
    X_seq.append(X[i : i+seq_length])
    y_seq.append(y[i+seq_length])

X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32)

# 划分训练集 (前80%数据) 和 测试集 (后20%数据)
train_size = int(len(X_seq) * 0.8)
X_train, y_train = X_seq[:train_size], y_seq[:train_size]
X_test, y_test = X_seq[train_size:], y_seq[train_size:]

# 4. 定义 LSTM 网络结构
class FC_LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(FC_LSTM, self).__init__()
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # 只取最后一个时间步的特征进行预测
        return out

model = FC_LSTM(input_size=4, hidden_size=64, num_layers=2)
criterion = nn.MSELoss() # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 模型训练
epochs = 600
print("开始训练 LSTM 模型...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')


# 6. 模型评估与画图 (保研PPT核心素材)
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test)
    
# 将预测结果反归一化，变回真实的 SOH 范围
y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
y_test_real = scaler_y.inverse_transform(y_test.numpy())

# 绘制 对比图
plt.figure(figsize=(10, 5))
plt.plot(y_test_real, label='Real SOH (Simulink Physical Model)', color='blue', linewidth=2)
plt.plot(y_pred, label='Predicted SOH (LSTM Model)', color='red', linestyle='--', linewidth=2)
plt.title('Fuel Cell SOH Degradation Prediction on Test Dataset')
plt.xlabel('Time Steps (Seconds)')
plt.ylabel('State of Health (SOH)')
plt.legend()
plt.grid(True)
plt.show()