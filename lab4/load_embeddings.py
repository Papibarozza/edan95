from joblib import load,dump
word_embedding_dict = {}
with open("glove.6B.100d.txt",encoding="utf8") as f:
  line = f.readline()
  #print(line)
  while(line):
    tab_separated = line.split()
    key = tab_separated[0]
    vals = list(map(float,tab_separated[1:]))
    word_embedding_dict[key] = vals
    line= f.readline()
  dump(word_embedding_dict,'word_embeddings')
    