import re
import string

import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
    tags = set(['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'])
    ret = []
    for word, pos in nltk.pos_tag(list_no_stopwords_stem):
        if pos not in tags:
            ret.append(word)
    # print("词性过滤之后结果:", ret)

    # 过滤后结果
    text_filtered = ' '.join(ret)  # ''.join() would join without spaces between words.
    #     print('过滤后:', text_filtered)
    return text_filtered

# eng_text_clean('Quiet location but so close to the London buzz.  Good restaurants and pubs nearby but do your research as many are closed on Sundays. Friendly efficient staff, great breakfast, had one of the best nights sleep Ive ever had in London ')
