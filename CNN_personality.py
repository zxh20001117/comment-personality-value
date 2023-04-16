import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F

from Function4Torch import make_train_dataset
from configparser import ConfigParser

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')


class CNNModel(nn.Module):
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
        self.conv_channel = conv_channel
        self.conv_nums = conv_nums
        self.linear_channel = linear_channel
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels=words_channels, out_channels=conv_channel, kernel_size=1, stride=1,
                               padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=words_channels, out_channels=conv_channel, kernel_size=2, stride=1,
                               padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=words_channels, out_channels=conv_channel, kernel_size=3, stride=1,
                               padding=0)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=conv_channel*3, out_channels=conv_channel*3, kernel_size=1, stride=1,
                               padding=0)

        self.maxpooling1 = nn.AdaptiveMaxPool1d(1)
        self.maxpooling2 = nn.AdaptiveMaxPool1d(1)
        self.maxpooling3 = nn.AdaptiveMaxPool1d(1)

        self.maxpooling4 = nn.AdaptiveMaxPool1d(1)

        self.linear1 = nn.Linear(in_features=conv_nums * conv_channel + ma_feats, out_features=linear_channel)
        self.act = nn.Sigmoid()
        self.linear2 = nn.Linear(in_features=linear_channel, out_features=out_channels)

        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x, ma=None):
        '''
        x : [batch_size, seq_nums, words_nums, words_channels]
        ma: [batch_size, ma_channels]
        '''
        b, s, w, c = x.shape
        x = x.reshape(b * s, w, c)

        x = x.permute(0, 2, 1)

        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(x))
        out3 = self.relu3(self.conv3(x))

        out1 = self.maxpooling1(out1).squeeze()
        out2 = self.maxpooling2(out2).squeeze()
        out3 = self.maxpooling3(out3).squeeze()

        out1 = out1.reshape(b, s, self.conv_channel)
        out2 = out2.reshape(b, s, self.conv_channel)
        out3 = out3.reshape(b, s, self.conv_channel)

        out = torch.cat((out1, out2, out3), dim=-1)

        out = out.permute(0, 2, 1)
        # out = self.conv4(out)
        out = self.maxpooling4(out).squeeze(2)
        out = torch.cat((out, ma), dim=-1)

        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.linear2(out)

        # out = F.softmax(out, dim=-1)
        return out

def train_personality_model(personality):
    model = CNNModel().cuda()
    best = 53
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # criterion = nn.NLLLoss()  # 负对数似 然
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)  # Adadelta梯度优化器

    train_loader, test_loader = make_train_dataset(personality)
    trainlog = []

    for epoch in range(200):
        loss_list = []
        for batch_x, batch_m, batch_y in train_loader:
            batch_x, batch_m, batch_y = batch_x.cuda(), batch_m.cuda(), batch_y.cuda()
            pred = model(batch_x, batch_m)
            loss = criterion(pred, batch_y)  # batch_y类标签就好，不用one-hot形式

            loss_list.append(float(f'{loss}'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(sum(loss_list)/len(loss_list)))

        test_acc_list = []
        test_loss_list = []
        test_correct = 0
        with torch.no_grad():
            for data, m, target in test_loader:
                data, m, target = data.cuda(), m.cuda(), target.cuda()
                output = model(data, m)

                loss = criterion(output, target)  # batch_y类标签就好，不用one-hot形式
                test_loss_list.append(float(f'{loss}'))

                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        # test_acc_list.append(100. * correct / len(test_loader.dataset))
        # print('test  Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),
        # 100. * correct / len(test_loader.dataset)))

        train_correct = 0
        with torch.no_grad():
            for data, m, target in train_loader:
                data, m, target = data.cuda(), m.cuda(), target.cuda()
                output = model(data, m)
                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                train_correct += pred.eq(target.view_as(pred)).sum().item()
        # train_acc_list.append(100. * correct / len(train_loader.dataset))
        # print('train Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(train_loader.dataset),
        #                                            100. * correct / len(train_loader.dataset)))
        # columns = ['epoch', 'train loss', 'test loss', 'train acc', 'test acc']
        batch_log = [epoch, sum(loss_list) / len(loss_list), sum(test_loss_list) / len(test_loss_list),
                         100. * train_correct / len(train_loader.dataset), 100. * test_correct / len(test_loader.dataset)]
        print(batch_log)
        trainlog.append(batch_log)

        if 100. * test_correct / len(test_loader.dataset) > best:
            best = 100. * test_correct / len(test_loader.dataset)
            print(f"{personality} epoch-{epoch} test acc:{best}")
            torch.save({'model': model.state_dict()},
                       f"{conf.get('model', 'modelPath')}/{personality} personality classification model.pth")

        if epoch %10 == 9:
            print(f"{personality} epoch-{epoch} finished")

    trainlog = pd.DataFrame(trainlog, columns=['epoch', 'train loss', 'test loss', 'train acc', 'test acc'])
    trainlog.to_excel(f"{conf.get('model', 'logPath')}/{personality} train log.xlsx")

if __name__ == "__main__":
    # personalities = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
    personalities = [ "cNEU", "cAGR", "cCON", "cOPN"]
    for personality in  personalities:
        train_personality_model(personality)

