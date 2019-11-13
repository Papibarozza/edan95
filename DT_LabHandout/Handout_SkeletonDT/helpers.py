import numpy as np
import ToyData as td
import math
def most_common(target):
    return max(set(target), key=target.count)

def entropy(target,classes):
  if(len(target) <=1):
    return 1
  else:
    counts = [0]*len(classes)
    for i,cl in enumerate(classes):
      for tar in target:
        if(cl == tar):
            counts[i] += 1
    probs = counts/np.sum(counts)
    entropy = 0
    for prob in probs:
      if(prob != 0):
        entropy-= prob*math.log2(prob)
      else:
        entropy-= 0
    return entropy

def information_gain(target,classes,data,attributes):
  entr = entropy(target,classes)
  nmbr_data_points = len(data)
  attr_entrop = {}
  for attr in attributes:
    curr_sum = 0
    for val in attributes[attr]:
      target_idxs = [ i for i in range(nmbr_data_points) if val in data[i] ]
      tar = np.array(list(target))
      curr_sum+= (len(target_idxs)/nmbr_data_points)*entropy(tuple(tar[target_idxs]),classes)
    attr_entrop[attr] = entr-curr_sum
    curr_sum = 0
  return attr_entrop

def get_best_split(target,classes,data,attributes):
   ig = information_gain(target,classes,data,attributes)
   return max(ig, key=lambda key: ig[key])


if __name__ == "__main__":
   attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
   ig = information_gain(target,classes,data,attributes)

   print(get_best_split(target,classes,data,attributes))