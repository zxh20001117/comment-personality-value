from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import time
import pandas as pd

from Function import process_user_personality_sentences, get_word2vev_vectors

# process_user_personality_sentences()

# data = pd.read_json('dataProcess/user filtered sentences.json')
#
# data['docVectors'] = data.apply(lambda x: get_word2vev_vectors(x['sentences']), axis= 1)

startTime = time.time()
pretrained_model = KeyedVectors.load_word2vec_format("word2vec model/GoogleNews-vectors-negative300.bin.gz",
                                                     binary=True)
modelLoadTime = time.time()
print(f'it takes {int(modelLoadTime - startTime)} s for loading word2vec model\n')

res = get_word2vev_vectors(['its always glad staying in shangrila pudong as the service here keeps the high standard', 'the manager kenny remembered me when i entered to the lounge since last meetup in last year', 'easier for my business trip', 'the breakfast was so local and delicious'], pretrained_model)
print(res[0])