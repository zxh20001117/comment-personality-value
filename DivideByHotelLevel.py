import pickle
import time

import pandas as pd
from gensim.models import KeyedVectors

from Function import content_slice2sentences, content2attribute_sentences, sentence_clean, get_emtional_words, \
    spilt_20_sentences, cal_run_time

record_time = time.time()
with open("dataProcess/value_attributes.pickle", "rb") as f:
    attribute_group = pickle.load(f)


data = pd.read_excel('dataProcess/hotel level merged.xlsx')
hotel_level = pd.read_excel('dataProcess/hotel_level_links.xlsx')
record_time = cal_run_time(record_time, "loading data")

emotional_words = get_emtional_words()
level_list = ['high', 'middle']
# pretrained_model =KeyedVectors.load_word2vec_format("word2vec model/GoogleNews-vectors-negative300.bin.gz",
#                                                      binary=True)

# wordDic = {pretrained_model.index_to_key[i]: 'i' for i in range(len(pretrained_model.index_to_key))}
# record_time = cal_run_time(record_time, "loading words and word2vec moodel")

for level in level_list:
    hotel_names = hotel_level[hotel_level['level'] == level]['hotelname']
    level_data = data[data['hotel_name'].isin(hotel_names)]
    level_data = level_data[['id_review', 'title', 'review', 'rating']]
    level_data['sentences'] = level_data.apply(lambda x: content_slice2sentences(f"{x['title']}. {x['review']}"), axis=1)

    record_time = cal_run_time(record_time, "slicing data")

    for values in attribute_group.keys():
        attribute_stems = {i: 0 for i in attribute_group[values]}
        level_data[f'{values}_sentences'] = level_data.apply(lambda x: content2attribute_sentences(
            x['sentences'], attribute_stems
            ), axis=1
        )
        record_time = cal_run_time(record_time, f"generating {level}-{values} sentences")

    level_data['sentences'] = level_data.apply(lambda x: sentence_clean(x['sentences']), axis=1)
    level_data['sentences'] = level_data.apply(lambda x: spilt_20_sentences(x['sentences']), axis=1)

    level_data.to_json(f'dataProcess/{level} level hotel data.json')
    record_time = cal_run_time(record_time, f"processing {level} personality data and storing data")
