import time
import pandas as pd
import pickle
import nltk

from Function import content2attribute_sentences

with open("dataProcess/value_attributes.pickle", "rb") as f:
    attribute_group = pickle.load(f)

startTime = time.time()
data = pd.read_excel('dataProcess/all merged.xlsx')
wordLoadTime = time.time()
print(f'it takes {wordLoadTime - startTime} s for loading data\n')

for values in attribute_group.keys():
    data[f'{values}_sentences'] = data.apply(lambda x: content2attribute_sentences(f"{x['title']}. {x['review']}", attribute_group[values]), axis=1)
wordProcessTime = time.time()
print(f'it takes {wordProcessTime - startTime} s for processing data\n')

result = data.drop(['rating_review',	'n_review_user'	,'n_votes_revie', 'location', 'user_name', 'user_link', 'date_of_stay', 'review'], axis = 1)
result.to_json('dataProcess/value attribute sentences.json')