import re
import string
import time

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

stoplist = list(pd.read_csv('stop_words_eng.txt', names=['w'], encoding='utf-8', engine='python').w)
cache_english_stopwords = stopwords.words('english') + stoplist


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


def content_slice2sentences(content):
    sentences = [s.strip().lower() for s in re.split('[,.!?]', content) if len(s.strip()) >= 10]
    return sentences


def group_reviews_by_userlink(data):
    data2 = data.groupby('user_link')['review'].apply(lambda x: x.str.cat(sep='.')).reset_index()
    data2.to_json('dataProcess/user grouped reviews.json')
    return data2


def chat_statistics(data):
    # print(data.sort_values(['len'], ascending=(False)))
    data.boxplot(column=['len'])
    plt.show()

    data['sentenceNum'] = data.apply(lambda x: len(x['sentences']), axis=1)
    # print(data.sort_values(['sentenceNum'], ascending=(False)))

    data.boxplot(column=['sentenceNum'])
    plt.show()
    print(data[['len', 'sentenceNum']].describe())


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


def filter_emotionless_sentences(sentences):
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
    res = []
    for sentence in sentences:
        flag = False
        for word in sentence.split():
            if word in emotional_words:
                flag = True
                break
        if flag:
            res.append(sentence)
    return res


def get_word2vev_vectors(sentences, model):
    w2v_word_list = list(model.index_to_key)
    maxSentencelen = 24
    maxWordCount = 20

    docVec = np.zeros((maxSentencelen, maxWordCount, 300))
    for i in range(min(len(sentences), maxSentencelen)):
        get_sentence_vectors(sentences[i], docVec[i], model, w2v_word_list)
    return docVec


def get_sentence_vectors(sentence, sentenceVector, model, w2v_word_list):
    maxWordCount = 20
    words = sentence.split()

    for i in range(min(len(words), maxWordCount)):
        if words[i] in w2v_word_list:
            sentenceVector[i] += model.get_vector(words[i])


def process_user_personality_sentences():
    startTime = time.time()
    data = pd.read_json('dataProcess/user grouped reviews.json')
    wordLoadTime = time.time()
    print(f'it takes {int(wordLoadTime - startTime)} s for loading words\n')

    data['len'] = data.apply(lambda x: len(x['review']), axis=1)
    data['sentences'] = data.apply(lambda x: content_slice2sentences(x['review']), axis=1)
    sliceTime = time.time()
    print(f'it takes {int(sliceTime - wordLoadTime)} s for slicing contents\n')

    print("\n原始数据中 按照user_link汇总之后 评论的各项统计数据：")
    chat_statistics(data)
    words_in_sentences_statistics(data)

    data['sentences'] = data.apply(lambda x: sentence_clean(x['sentences']), axis=1)
    cleanTime = time.time()
    print(f'it takes {int(cleanTime - sliceTime)} s for cleaning contents\n')

    data['sentences'] = data.apply(lambda x: filter_emotionless_sentences(x['sentences']), axis=1)
    filterTime = time.time()
    print(f'it takes {int(filterTime - cleanTime)} s for filtering contents\n')
    # data.to_json('dataProcess/user filtered sentences.json')
    # data.to_excel('dataProcess/user filtered sentences.xlsx')

    del data
    data = pd.read_json('dataProcess/user filtered sentences.json')
    print("\n根据人格分析预处理规则处理后 user_link汇总 评论的各项统计数据：")
    chat_statistics(data)
    words_in_sentences_statistics(data)

    data['sentences'] = data.apply(lambda x: spilt_20_sentences(x['sentences']), axis=1)
    # data = data[data['sentences'].map(len) > 0]
    print("\n每一句最长20单词处理之后 评论的各项统计数据：")
    chat_statistics(data)
    words_in_sentences_statistics(data)

def spilt_20_sentences(sentences):
    maxWordCount = 20
    res = []
    for sentence in sentences:
        words = sentence.split()
        num = len(words)
        for i in range(num // maxWordCount):
            res.append(" ".join(words[i*maxWordCount:(i+1)*maxWordCount]))
        res.append(" ".join(words[(num // maxWordCount)*maxWordCount:]))
    return res
