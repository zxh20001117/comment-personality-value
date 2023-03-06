import torch
import torch.nn as nn

import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, seq_nums=17,
                 words_nums=50, words_channels=300, ma_feats=84,
                 conv_channel=200, conv_nums=3, linear_channel=200, out_channels=2):
        super().__init__()
        self.seq_nums = seq_nums
        self.words_nums = words_nums
        self.words_channels = words_channels
        self.ma_feats = ma_feats

        self.lstm_num = 300
        self.lstm_layer = 2
        self.linear_channel = linear_channel
        self.out_channels = out_channels
        self.lstm_filter = nn.LSTM(
            #                                     input_size = seq_nums*words_nums,
            input_size=words_channels,
            hidden_size=self.lstm_num,
            bidirectional=True,
            batch_first=True
        )

        self.linear1 = nn.Linear(in_features=self.lstm_layer * self.lstm_num + ma_feats, out_features=linear_channel)
        self.act = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(in_features=linear_channel, out_features=out_channels)

    def forward(self, x, ma=None):
        '''
        x : [batch_size, seq_nums, words_nums, words_channels]
        ma: [batch_size, ma_channels]
        '''
        b, s, w, c = x.shape
        x = x.reshape(b, s * w, c)

        out, (_, _) = self.lstm_filter(x)
        output = out[:, -1, :].squeeze()

        out = torch.cat((output, ma), dim=-1)

        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.act(out)

        out = self.linear2(out)

        out = F.softmax(out, dim=-1)

        return out


if __name__ == "__main__":
    model = LSTMModel().cuda()
    x = torch.randn(2, 17, 50, 300).cuda()
    ma = torch.randn(2, 84).cuda()

    out = model(x, ma)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)  # Adadelta梯度优化器

    loss = criterion(out, torch.tensor([1, 1]).cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for epoch in range(200):
        pred = model(x, ma)
        loss = criterion(pred, torch.tensor([1, 1]).cuda())  # batch_y类标签就好，不用one-hot形式
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%4d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

