import time
import pandas as pd
import pickle

from Function import content2attribute_sentences, content_slice2sentences

with open("dataProcess/value_attributes.pickle", "rb") as f:
    attribute_group = pickle.load(f)

startTime = time.time()
data = pd.read_excel('dataProcess/all merged.xlsx')
wordLoadTime = time.time()

print(data.keys())
data['sentences'] = data.apply(lambda x: content_slice2sentences(f"{x['title']}. {x['review']}"), axis=1)
for values in attribute_group.keys():
    attribute_stems = {i : 0 for i in attribute_group[values]}
    data[f'{values}_sentences'] = data.apply(lambda x: content2attribute_sentences(
        x['sentences'], attribute_stems
    )
                                             , axis=1)
wordProcessTime = time.time()
print(f'it takes {wordProcessTime - startTime} s for processing data\n')

result = data.drop(['rating_review', 'n_review_user',
                    'n_votes_review', 'location',
                    'user_name', 'user_link',
                    'date_of_stay', 'review',
                    'sentences']
                   , axis=1)
result.to_json('dataProcess/value attribute sentences 0326.json')
