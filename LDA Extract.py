import pandas as pd
import time

import pyLDAvis.gensim_models

from Function import eng_text_clean
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models

# data = pd.read_excel('dataProcess/all merged.xlsx')
# starttime = time.time()
# data['words'] = data.apply(lambda x: eng_text_clean(x['review']).split(), axis = 1)
# print(f'it takes {time.time() - starttime} s for 44w comments')
# data['words'].to_excel('dataProcess/LDA Processed.xlsx')


wordsData = pd.read_excel("dataProcess/LDA Processed.xlsx")
chapList = [eval(i) for i in wordsData['words']]

# 构建词典
dictionary = corpora.Dictionary(chapList)
# 根据词典转换为词袋向量
corpus = [dictionary.doc2bow(text) for text in chapList]  # 仍为list in list

tfidf_model = models.TfidfModel(corpus)  # 建立TF-IDF模型
corpus_tfidf = tfidf_model[corpus]  # 对所需文档的词袋向量计算TF-IDF结果

# 将词典与文档的词袋向量放入LDA模型进行训练

num_topics = 10  # 主题个数
passes = 10
ldamodel1 = LdaModel(corpus
                     , id2word=dictionary
                     , num_topics=num_topics  # 主题个数
                     , passes=passes  # 训练过程中穿过语料库的次数 对于大量语料 需要响应变大
                     )

# LDA结果的可视化工具 产生一个html 因为使用到了外网的css资源 需要翻墙才能显示
vis = pyLDAvis.gensim_models.prepare(ldamodel1, corpus, dictionary)
pyLDAvis.save_html(vis, 'LDA Visual/topics{}_passes{}.html'.format(num_topics, passes))
