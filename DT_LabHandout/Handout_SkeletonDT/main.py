import ToyData as td
import ID3
import math
import numpy as np
from sklearn import tree, metrics, datasets
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ["PATH"] += os.pathsep +'C:/Users/Arvid/Anaconda3/envs/lab2/Library/bin/graphviz/'



      
def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

    id3 = ID3.ID3DecisionTreeClassifier()
    #print(attributes)
    myTree = id3.fit(data, target, attributes, classes)
    
    graph = id3.make_dot_data()
    graph.render("mytreetest")


if __name__ == "__main__": main()