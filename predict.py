import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from gensim.models import KeyedVectors

from CNN_personality import CNNModel
from Function import get_word2vev_vectors, cal_run_time

from configparser import ConfigParser

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')


personalities = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
models = {}
pretrained_model = KeyedVectors.load_word2vec_format("word2vec model/GoogleNews-vectors-negative300.bin.gz",
                                                     binary=True)
wordDic = {pretrained_model.index_to_key[i]: 'i' for i in range(len(pretrained_model.index_to_key))}

for i in personalities:
    models[i] = CNNModel().cuda()
    models[i].load_state_dict(torch.load(f"model/{i} personality classification model.pth")['model'])

result = {}
for i in personalities:
    result[i] = []


def predict_5_personalities(sentences, LIWC, index):
    data = pd.DataFrame()
    data['docVector'] = sentences.apply(lambda x: get_word2vev_vectors(x['sentences'], pretrained_model, wordDic), axis=1)
    docVector = np.stack([i for i in data['docVector']], axis=0)
    docVector = torch.FloatTensor(docVector)
    input_liwc = np.array(LIWC)
    input_liwc = torch.FloatTensor(input_liwc)
    predict_dataset = Data.TensorDataset(docVector, input_liwc)
    predict_loader = Data.DataLoader(
        dataset=predict_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=conf.getint('predict', 'batch'),  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
        drop_last=False,  # 要不要丢弃最后不足一个batch的数据
    )
    recordTime = time.time()

    for i in personalities:
        for x, m in predict_loader:
            models[i].eval()
            output = models[i](x.cuda(), m.cuda())
            probs = torch.softmax(output, dim=1)
            true_probs = probs[:, 1].tolist()
            result[i] = result[i] + true_probs
        recordTime = cal_run_time(recordTime, f"{index}/{nGroup} group predict {i} personality")
    del data, docVector, input_liwc, predict_dataset, predict_loader

if __name__ == '__main__':
    level = 'middle'
    data = pd.read_json(f"dataProcess/{level} level hotel data.json")
    LIWC = pd.read_csv(f"dataProcess/{level} level hotel reviews - LIWC Analysis.csv")

    nGroup = 20
    data_groups = []
    LIWC_groups = []
    length = len(data)//nGroup

    for i in range(nGroup - 1):
        data_groups.append(data.iloc[i*length:(i+1)*length].copy())
        LIWC_groups.append(LIWC.iloc[i*length:(i+1)*length].copy())
    data_groups.append(data.iloc[(nGroup-1)*length:].copy())
    LIWC_groups.append(LIWC.iloc[(nGroup-1)*length:].copy())
    del data


    for index, t in enumerate(data_groups):
        predict_5_personalities(data_groups[index], LIWC_groups[index], index)

    pd.DataFrame.from_dict(result).to_csv(f"result/{level} level hotel personality predict.csv", index=False)
