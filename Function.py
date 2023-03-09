import random
import re
import string
import time

import torch
import torch.utils.data as Data

import nltk
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from configparser import ConfigParser

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')

stoplist = list(pd.read_csv('stop_words_eng.txt', names=['w'], encoding='utf-8', engine='python').w)
cache_english_stopwords = stopwords.words('english') + stoplist
UNKVEC = np.load("Global UNKVEC.npy")


def eng_text_clean(text):
    #     print('原始数据:', text, '\n')
    # 去掉一些价值符号
    text_no_tickers = re.sub(r'\$\w*', '', text)
    # print('去掉价值符号后的:', text_no_tickers, '\n')

    # 去掉超链接
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', ' ', text_no_tickers)
    # print('去掉超链接后的:', text_no_hyperlinks, '\n')

    # 去掉一些专门名词缩写，简单来说就是字母比较少的词
    text_no_small_words = re.sub(r'\b\w{1,2}\b', '', text_no_hyperlinks)
    # print('去掉专门名词缩写后:', text_no_small_words, '\n')

    # 去除标点符号
    remove = str.maketrans(string.punctuation, "{:<32}".format(""))
    text_no_small_words = text_no_small_words.translate(remove)
    # print('去除标点:', text_no_small_words)
    text_no_small_words = re.sub(r'[^A-Za-z0-9 ]+', ' ', text_no_small_words)

    # 去掉多余的空格
    text_no_whitespace = re.sub(r'\s\s+', ' ', text_no_small_words)
    text_no_whitespace = text_no_whitespace.lstrip(' ')
    # print('去掉空格后的:', text_no_whitespace, '\n')

    # 分词
    tokens = word_tokenize(text_no_whitespace.lower())
    # print('分词结果:', tokens, '\n')

    # 去停用词
    porter_stemmer = PorterStemmer()
    list_no_stopwords_stem = [porter_stemmer.stem(i) for i in tokens if i not in cache_english_stopwords and len(i) > 1]

    # 去除 否定词、情感度词和情感词
    tags = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', "NEG"}
    ret = []
    for word, pos in nltk.pos_tag(list_no_stopwords_stem):
        if pos not in tags:
            ret.append(word)
    # print("词性过滤之后结果:", ret)

    # 过滤后结果
    text_filtered = ' '.join(ret)  # ''.join() would join without spaces between words.
    #     print('过滤后:', text_filtered)
    return text_filtered


def content2attribute_sentences(sentences, stem_list):
    porter_stemmer = PorterStemmer()
    res = []
    for sentence in sentences:
        for word in word_tokenize(sentence):
            if porter_stemmer.stem(word) in stem_list:
                text_no_small_words = re.sub(r'\b\w{1,2}\b', '', sentence)

                remove = str.maketrans(string.punctuation, "{:<32}".format(""))
                text_no_small_words = text_no_small_words.translate(remove)
                text_no_small_words = re.sub(r'[^A-Za-z0-9 ]+', ' ', text_no_small_words)

                text_no_whitespace = re.sub(r'\s\s+', ' ', text_no_small_words)
                text_no_whitespace = text_no_whitespace.lstrip(' ')
                res.append(text_no_whitespace)
                break
    return res


def content_slice2sentences(content, include_comma = True):
    if include_comma:
        sentences = [s.strip().lower() for s in re.split('[,.!?]', content) if len(s.strip()) >= 10]
    else:
        sentences = [s.strip().lower() for s in re.split('[.!?]', content) if len(s.strip()) >= 10]
    return sentences


def group_reviews_by_userlink(data):
    data2 = data.groupby('user_link')['review'].apply(lambda x: x.str.cat(sep='.')).reset_index()
    data2.to_json('dataProcess/user grouped reviews.json')
    return data2


def chat_statistics(data):
    # print(data.sort_values(['len'], ascending=(False)))

    data['sentenceNum'] = data.apply(lambda x: len(x['sentences']), axis=1)
    # print(data.sort_values(['sentenceNum'], ascending=(False)))

    data.boxplot(column=['sentenceNum'])
    plt.show()
    print(data[['sentenceNum']].describe())


def sentence_clean(sentences):
    res = []
    for sentence in sentences:
        pattern = r"[^a-zA-Z0-9\s\'\"]"
        clean_sentence = re.sub(pattern, "", sentence)
        res.append(clean_sentence)
    return res


def words_in_sentences_statistics(data):
    templist = []
    for i in data['sentences']:
        for j in i:
            templist.append(len(j.split()))
    tempdf = pd.DataFrame(templist, columns=['wordNum'])
    tempdf.boxplot()
    plt.show()


def get_emtional_words():
    emotion_lexion = pd.read_csv('Emotion_Lexicon.csv')
    emotional_words = emotion_lexion[(emotion_lexion['anger'] == 1) |
                                     (emotion_lexion['anticipation'] == 1) |
                                     (emotion_lexion['disgust'] == 1) |
                                     (emotion_lexion['fear'] == 1) |
                                     (emotion_lexion['joy'] == 1) |
                                     (emotion_lexion['negative'] == 1) |
                                     (emotion_lexion['positive'] == 1) |
                                     (emotion_lexion['sadness'] == 1) |
                                     (emotion_lexion['surprise'] == 1) |
                                     (emotion_lexion['trust'] == 1)
                                     ]['Words']
    emotional_words = emotional_words.tolist()
    emotional_words = {emotional_words[i]: i for i in range(len(emotional_words))}
    return emotional_words


def filter_emotionless_sentences(sentences, emotional_words):
    res = []
    for sentence in sentences:
        flag = False
        for word in sentence.split():
            if emotional_words.get(word) is not None:
                flag = True
                break
        if flag:
            res.append(sentence)
    return res


def get_word2vev_vectors(sentences, model, w2v_word_dic):
    maxSentenceLen = conf.getint("model", "seq_nums")
    maxWordCount = conf.getint("model", "words_nums")

    docVec = np.zeros((maxSentenceLen, maxWordCount, 300))
    for i in range(min(len(sentences), maxSentenceLen)):
        get_sentence_vectors(sentences[i], docVec[i], model, w2v_word_dic)
    for i in range(len(sentences), maxSentenceLen):
        for j in range(maxWordCount):
            docVec[i][j] -= np.zeros(300)
    return docVec


def get_sentence_vectors(sentence, sentence_vector, model, w2v_word_dic):
    maxWordCount = conf.getint("model", "words_nums")
    words = sentence.split()

    for i in range(min(len(words), maxWordCount)):
        if w2v_word_dic.get(words[i]) is not None:
            sentence_vector[i] += model.get_vector(words[i])
        else:
            sentence_vector[i] += UNKVEC
    for i in range(len(words), maxWordCount):
        sentence_vector[i] -= np.zeros(300)


def process_user_personality_sentences():
    startTime = time.time()
    data = pd.read_json('dataProcess/user grouped reviews.json')
    wordLoadTime = time.time()
    print(f'it takes {int(wordLoadTime - startTime)} s for loading words\n')

    data['len'] = data.apply(lambda x: len(x['review']), axis=1)
    data['sentences'] = data.apply(lambda x: content_slice2sentences(x['review'], include_comma=False), axis=1)
    sliceTime = time.time()
    print(f'it takes {int(sliceTime - wordLoadTime)} s for slicing contents\n')

    print("\n原始数据中 按照user_link汇总之后 评论的各项统计数据：")
    chat_statistics(data)
    words_in_sentences_statistics(data)

    data['sentences'] = data.apply(lambda x: sentence_clean(x['sentences']), axis=1)
    cleanTime = time.time()
    print(f'it takes {int(cleanTime - sliceTime)} s for cleaning contents\n')

    emotional_words = get_emtional_words()
    data['sentences'] = data.apply(lambda x: filter_emotionless_sentences(x['sentences'], emotional_words), axis=1)
    filterTime = time.time()
    print(f'it takes {int(filterTime - cleanTime)} s for filtering contents\n')

    print("\n根据人格分析预处理规则处理后 user_link汇总 评论的各项统计数据：")
    chat_statistics(data)
    words_in_sentences_statistics(data)

    data['sentences'] = data.apply(lambda x: spilt_20_sentences(x['sentences']), axis=1)
    data = data[data['sentences'].map(len) > 0]
    print("\n每一句最长20单词处理之后 评论的各项统计数据：")
    chat_statistics(data) # 每一个人 上限 18句
    words_in_sentences_statistics(data)

    # data.to_json('dataProcess/user filtered sentences.json')
    # data.to_excel('dataProcess/user filtered sentences.xlsx')


def spilt_20_sentences(sentences):
    maxWordCount = 20
    res = []
    for sentence in sentences:
        words = sentence.split()
        num = len(words)
        if num < 30:
            res.append(sentence)
            continue
        group_num = num // maxWordCount
        remainder = num % maxWordCount
        for i in range(group_num):
            res.append(" ".join(words[i * maxWordCount:(i + 1) * maxWordCount]))
        if remainder>0:
            res.append(" ".join(words[group_num * maxWordCount:]))
    return res


def process_train_data(pretrained_model):
    data = pd.read_csv('dataProcess/essays.csv')
    data = data.drop(['#AUTHID'], axis=1)

    f_names = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    for i in f_names:
        label = preprocessing.LabelEncoder()
        data[i] = label.fit_transform(data[i])  # 数据标准化

    startTime = time.time()
    data['sentences'] = data.apply(lambda x: content_slice2sentences(x['TEXT'], include_comma=False), axis=1)
    # print(data.iloc[0, 6])
    data['sentences'] = data.apply(lambda x: sentence_clean(x['sentences']), axis=1)
    # print(data.iloc[0, 6])
    sentenceGenTime = time.time()
    print(f'it takes {int(sentenceGenTime - startTime)} s for preprocessing\n')

    emotional_words = get_emtional_words()
    data['sentences'] = data.apply(lambda x: spilt_20_sentences(x['sentences']), axis=1)
    data['sentences'] = data.apply(lambda x: filter_emotionless_sentences(x['sentences'], emotional_words), axis=1)
    data['sentences'] = data.apply(lambda x: delete_half_sentences(x['sentences']), axis=1)

    preprocessingTime = time.time()
    print(f'it takes {int(preprocessingTime - sentenceGenTime)} s for filtering\n')
    # print(data.iloc[0, 6])
    print("\n 训练集处理之后 评论的各项统计数据：")
    chat_statistics(data)
    words_in_sentences_statistics(data)

    wordDic = {pretrained_model.index_to_key[i]: 'i' for i in range(len(pretrained_model.index_to_key))}
    data['docVector'] = data.apply(lambda x: get_word2vev_vectors(x['sentences'], pretrained_model, wordDic), axis=1)
    vectorGeneratingTime = time.time()
    print(f'it takes {int(vectorGeneratingTime - preprocessingTime)} s for generating doc vectors\n')

    docVector = np.stack([i for i in data['docVector']], axis=0)
    np.save('dataProcess/train data docVector.npy', docVector)
    print(docVector.shape)

    data = data.drop(['TEXT', 'sentenceNum', 'docVector', 'sentences'], axis=1)
    data.to_json('dataProcess/train data label.json')



def delete_half_sentences(sentences):
    res = []
    # 计算要删除的元素个数
    delete_count = len(sentences) // 2

    # 生成要删除的元素的索引
    delete_indexes = random.sample(range(len(sentences)), delete_count)

    # 根据索引删除元素
    for index in range(len(sentences)):
        if index not in delete_indexes:
            res.append(sentences[index])
    return res


def make_train_dataset(label):
    labelData = pd.read_json('dataProcess/train data label.json')[label]
    docVec = np.load('dataProcess/train data docVector.npy')
    liwcFeatures = pd.read_csv('dataProcess/train LIWC-22 Results.csv')
    from sklearn import preprocessing
    zscore = preprocessing.StandardScaler()
    liwcFeatures = zscore.fit_transform(liwcFeatures)

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
        num_workers=4,  # 多进程（multiprocess）来读数据
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=4,  # 多进程（multiprocess）来读数据
    )
    return train_loader, test_loader



