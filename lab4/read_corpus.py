
from conll_dictorizer import CoNLLDictorizer, Token
from datasets import load_conll2003_en
from joblib import load,dump
import numpy as np

def extract_words_and_ner(sentence_dict):
  x = [None]*len(sentence_dict)
  y = [None]*len(sentence_dict)
  for i,elem in enumerate(sentence_dict):
    x[i] = elem['form']
    y[i] = elem['ner']
  return x,y


def create_vocabulary(sentence_dict,glove_embeddings,ner_tags):
  words = set()
  for elem in sentence_dict:
    word = elem[0]['form'].lower()
    if(word in glove_embeddings):
      words.add(word)
  for word in glove_embeddings:
    words.add(word)
  
  #9 indexes for the word tags after the first two
  wordlist = ['PAD','UNK']+ner_tags+list(words)
  #Create dictionary with words in wordlist and their indexes
  return  {k: v for v, k in enumerate(wordlist)}

def build_embedding_mtrx(vocabulary,embeddings):
  mtrx = np.random.rand(len(vocabulary),100)
  for word in vocabulary:
    mtrx[vocabulary[word],:] = embeddings[word]
  





if __name__ == "__main__":
  ner_tags = ['I-LOC','I-PER','I-ORG','I-MISC','B-LOC','B-PER','B-ORG','B-MISC','O']
  train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()
  conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
  train_dict = conll_dict.transform(train_sentences)
  #embeddings = load('word_embeddings.pkl')
  #print(embeddings)
  vocabulary = create_vocabulary(train_dict,load('word_embeddings.pkl'),ner_tags)
  #vocabulary = load('vocabulary.pkl')

