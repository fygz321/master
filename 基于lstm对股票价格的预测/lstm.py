import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)

def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

true_data = pd.read_csv('C:/Users/hbxfz/Desktop/算法设计与分析/LSTM/NFLX.csv')
dates = pd.to_datetime(true_data['Date'])  # 提取日期信息
target = 'High'

true_data = np.array(true_data[target])

test_data_size = 32
test_size = 0.15
train_size = 0.85
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))
train_data = true_data[:int(train_size * len(true_data))]
test_data = true_data[-int(test_size * len(true_data)):]
print("训练集尺寸:", len(train_data))
print("测试集尺寸:", len(test_data))
train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)

def create_inout_sequences(input_data, tw, pre_len, dates):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + 4) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        date_label = dates.iloc[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label, date_label))
    return inout_seq

pre_len = 4
train_window = 16
train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, dates)

class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=350, output_dim=1):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        h0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)
        c0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0_lstm, c0_lstm))
        out = out[:, -1]
        out = self.fc(out)

        return out

lstm_model = LSTM(input_dim=1, output_dim=pre_len, hidden_dim=train_window)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
epochs = 10
Train = False

if Train:
    losss = []
    lstm_model.train()
    start_time = time.time()
    for i in range(epochs):
        for seq, labels, _ in train_inout_seq:
            lstm_model.train()
            optimizer.zero_grad()

            y_pred = lstm_model(seq)

            single_loss = loss_function(y_pred, labels)

            single_loss.backward()
            optimizer.step()
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            losss.append(single_loss.detach().numpy())
    torch.save(lstm_model.state_dict(), 'save_model.pth')
    print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")
    plt.plot(losss)
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.savefig('training_error.png')
else:
    lstm_model.load_state_dict(torch.load('save_model.pth'))
    lstm_model.eval()
    results = []
    reals = []
    losss = []
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, dates)
    for seq, labels, date_labels in test_inout_seq:
        pred = lstm_model(seq)[0].item()
        results.append((date_labels.iloc[-1], pred))
        mae = calculate_mae(pred, labels.detach().numpy())
        reals.append(labels.detach().numpy())
        losss.append(mae)

    results_df = pd.DataFrame(results, columns=['Date', 'Forecast'])
    print(results_df)

    plt.style.use('ggplot')
    plt.plot(results_df['Date'], results_df['Forecast'], label='forecast', color='red', linestyle='--')
    plt.plot(results_df['Date'], reals, label='real', color='blue')  # 修改这一行
    plt.grid(True)
    plt.title('Real vs Forecast')  # 修改这一行
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.legend()
    plt.savefig('test_results.png')
    plt.show()