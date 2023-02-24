import pandas as pd
import time

import pyLDAvis.gensim_models

from Function import eng_text_clean
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models

data = pd.read_excel('dataProcess/all merged.xlsx')

startTime = time.time()

data['words'] = data.apply(lambda x: eng_text_clean(x['review']).split(), axis = 1)

data['words'].to_excel('dataProcess/LDA Processed 0224.xlsx')
wordProcessTime = time.time()
print(f'it takes {wordProcessTime - startTime} s for 44w comments\n')

wordsData = pd.read_excel("dataProcess/LDA Processed 0224.xlsx")
chapList = [eval(i) for i in wordsData['words']]
wordLoadTime = time.time()
print(f'it takes {wordLoadTime - wordProcessTime} s for loading words in LDA\n')

# 构建词典
dictionary = corpora.Dictionary(chapList)
# 根据词典转换为词袋向量
corpus = [dictionary.doc2bow(text) for text in chapList]  # 仍为list in list

tfidf_model = models.TfidfModel(corpus)  # 建立TF-IDF模型
corpus_tfidf = tfidf_model[corpus]  # 对所需文档的词袋向量计算TF-IDF结果

# 将词典与文档的词袋向量放入LDA模型进行训练

num_topics = 5  # 主题个数
passes = 15
ldaModel = LdaModel(corpus
                     , id2word=dictionary
                     , num_topics=num_topics  # 主题个数
                     , passes=passes  # 训练过程中穿过语料库的次数 对于大量语料 需要响应变大
                     )

# LDA结果的可视化工具 产生一个html 因为使用到了外网的css资源 需要翻墙才能显示
LDATime = time.time()
print(f'it takes {LDATime - wordLoadTime} s for generating LDA topics\n')

vis = pyLDAvis.gensim_models.prepare(ldaModel, corpus, dictionary)
pyLDAvis.save_html(vis, 'LDA Visual/topics{}_passes{}.html'.format(num_topics, passes))

topic_list = ldaModel.print_topics(num_topics, num_words = 15)
print(topic_list)
