import pickle

import pandas as pd

from DocVectorGenerate import pretrained_model
from Function import content_slice2sentences, content2attribute_sentences, sentence_clean, get_emtional_words, \
    spilt_20_sentences

with open("dataProcess/value_attributes.pickle", "rb") as f:
    attribute_group = pickle.load(f)


data = pd.read_excel('dataProcess/all merged.xlsx')
hotel_level = pd.read_excel('dataProcess/hotel level.xlsx')
emotional_words = get_emtional_words()
level_list = ['high', 'middle', 'low']
wordDic = {pretrained_model.index_to_key[i]: 'i' for i in range(len(pretrained_model.index_to_key))}

for level in level_list:
    hotel_names = hotel_level[hotel_level['level'] == level]['hotel names']
    level_data = data[data['level'].isin(hotel_names)]
    level_data = level_data[['id_review', 'title', 'review'], 'rating']
    level_data['sentences'] = level_data.apply(lambda x: content_slice2sentences(f"{x['title']}. {x['review']}"), axis=1)

    for values in attribute_group.keys():
        attribute_stems = {i: 0 for i in attribute_group[values]}
        level_data[f'{values}_sentences'] = level_data.apply(lambda x: content2attribute_sentences(
            x['sentences'], attribute_stems
            )
        )

    level_data['sentences'] = level_data.apply(lambda x: sentence_clean(x['sentences']), axis=1)
    level_data['sentences'] = level_data.apply(lambda x: spilt_20_sentences(x['sentences']), axis=1)

    level_data.to_json(f'dataProcess/{level} level hotel data.json')

