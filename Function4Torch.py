import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def make_train_dataset(label, LIWC_norm=False):
    labelData = pd.read_json('dataProcess/train data label.json')[label]
    docVec = np.load('dataProcess/train data docVector.npy')
    liwcFeatures = pd.read_csv('dataProcess/train LIWC-22 Results.csv')
    if LIWC_norm:
        zscore = preprocessing.StandardScaler()
        liwcFeatures = zscore.fit_transform(liwcFeatures)
    else:
        liwcFeatures = liwcFeatures.values

    input_vector = torch.FloatTensor(docVec)
    input_liwc = torch.FloatTensor(liwcFeatures)
    output_label = torch.LongTensor(labelData)

    x_train, x_test, m_train, m_test, y_train, y_test = train_test_split(input_vector, input_liwc, output_label, test_size=0.2,random_state = 0)

    train_dataset = Data.TensorDataset(torch.tensor(x_train), torch.tensor(m_train), torch.tensor(y_train))
    test_dataset = Data.TensorDataset(torch.tensor(x_test), torch.tensor(m_test), torch.tensor(y_test))

    batch_size = 4
    train_loader = Data.DataLoader(
        dataset=train_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )
    return train_loader, test_loader