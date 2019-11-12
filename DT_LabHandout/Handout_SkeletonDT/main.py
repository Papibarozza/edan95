import ToyData as td
import ID3

import numpy as np
from sklearn import tree, metrics, datasets
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

    id3 = ID3.ID3DecisionTreeClassifier()

    myTree = id3.fit(data, target, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(data2, myTree)


if __name__ == "__main__": main()