from sklearn.metrics import pairwise
from joblib import load,dump
import math

embeddings = load('word_embeddings.pkl')
print('Done with embeddings..')
def closest_match(word,dictionary):
  embedded_word = dictionary[word] if word in dictionary else Exception("No embedding available for word")
  temp_dict = { key:pairwise.cosine_similarity([dictionary[key]],[embedded_word])[0] for key in dictionary}
  for i in range(0,6):
    min_key = min(temp_dict.keys(), key=lambda k: 1.0-temp_dict[k])
    print(min_key,temp_dict[min_key])
    del temp_dict[min_key]

print('Closest match table: ')
closest_match('table',embeddings)
print('Closest match france: ')
closest_match('france',embeddings)
print('Closest match sweden: ')
closest_match('sweden',embeddings)