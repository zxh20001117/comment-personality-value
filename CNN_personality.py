import torch
import torch.nn as nn

import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, seq_nums=17,
                 words_nums=50, words_channels=300, ma_feats=84,
                 conv_channel=200, conv_nums=3, linear_channel=200, out_channels=2):
        super().__init__()
        self.seq_nums = seq_nums
        self.words_nums = words_nums
        self.words_channels = words_channels
        self.ma_feats = ma_feats
        self.conv_channel = conv_channel
        self.conv_nums = conv_nums
        self.linear_channel = linear_channel
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels=words_channels, out_channels=conv_channel, kernel_size=1, stride=1,
                               padding=0)
        self.conv2 = nn.Conv1d(in_channels=words_channels, out_channels=conv_channel, kernel_size=2, stride=1,
                               padding=0)
        self.conv3 = nn.Conv1d(in_channels=words_channels, out_channels=conv_channel, kernel_size=3, stride=1,
                               padding=0)

        self.maxpooling1 = nn.AdaptiveMaxPool1d(1)
        self.maxpooling2 = nn.AdaptiveMaxPool1d(1)
        self.maxpooling3 = nn.AdaptiveMaxPool1d(1)

        self.maxpooling4 = nn.AdaptiveMaxPool1d(1)

        self.linear1 = nn.Linear(in_features=conv_nums * conv_channel + ma_feats, out_features=linear_channel)
        self.act = nn.Sigmoid()
        self.linear2 = nn.Linear(in_features=linear_channel, out_features=out_channels)

    def forward(self, x, ma=None):
        '''
        x : [batch_size, seq_nums, words_nums, words_channels]
        ma: [batch_size, ma_channels]
        '''
        b, s, w, c = x.shape
        x = x.reshape(b * s, w, c)

        x = x.permute(0, 2, 1)

        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out1 = self.maxpooling1(out1).squeeze()
        out2 = self.maxpooling1(out2).squeeze()
        out3 = self.maxpooling1(out3).squeeze()

        out1 = out1.reshape(b, s, self.conv_channel)
        out2 = out2.reshape(b, s, self.conv_channel)
        out3 = out3.reshape(b, s, self.conv_channel)

        out = torch.cat((out1, out2, out3), dim=-1)

        out = out.permute(0, 2, 1)
        out = self.maxpooling4(out).squeeze()

        out = torch.cat((out, ma), dim=-1)

        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        out = F.softmax(out, dim=-1)

        return out

if __name__ == "__main__":
    model = CNNModel().cuda()
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