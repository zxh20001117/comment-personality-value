import torch
import torch.nn as nn

import torch.nn.functional as F

from Function4Torch import make_train_dataset
from configparser import ConfigParser

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')


class LSTMModel(nn.Module):
    def __init__(self, seq_nums=conf.getint("model", "seq_nums"),
                 words_nums=conf.getint("model", "words_nums"),
                 words_channels=conf.getint("model", "words_channels"),
                 ma_feats=conf.getint("model", "ma_feats"),
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
        output = out[:, -1, :]
        out = torch.cat((output, ma), dim=-1)

        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.act(out)

        out = self.linear2(out)

        out = F.softmax(out, dim=-1)

        return out


if __name__ == "__main__":
    model = LSTMModel().cuda()

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)  # Adadelta梯度优化器

    train_loader, test_loader = make_train_dataset('cEXT')
    for epoch in range(350):
        loss_list = []
        for batch_x, batch_m, batch_y in train_loader:
            batch_x, batch_m, batch_y = batch_x.cuda(), batch_m.cuda(), batch_y.cuda()
            pred = model(batch_x, batch_m)
            loss = criterion(pred, batch_y)  # batch_y类标签就好，不用one-hot形式

            loss_list.append(float(f'{loss}'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(sum(loss_list) / len(loss_list)))

            test_acc_list = []
            test_loss = 0
            correct = 0
            for data, m, target in test_loader:
                data, m, target = data.cuda(), m.cuda(), target.cuda()
                output = model(data, m)
                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct += pred.eq(target.view_as(pred)).sum().item()
            test_acc_list.append(100. * correct / len(test_loader.dataset))
            print('test  Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),

                                                           100. * correct / len(test_loader.dataset)))
            correct = 0
            train_acc_list = []
            for data, m, target in train_loader:
                data, m, target = data.cuda(), m.cuda(), target.cuda()
                output = model(data, m)
                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct += pred.eq(target.view_as(pred)).sum().item()
            train_acc_list.append(100. * correct / len(train_loader.dataset))
            print('train Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(train_loader.dataset),
                                                             100. * correct / len(train_loader.dataset)))

