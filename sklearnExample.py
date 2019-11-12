from sklearn.datasets import load_digits
from sklearn import tree
import graphviz
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import tree, metrics, datasets
import os
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from graphviz import Digraph

def render_tree(clf,name="mytree"):
  tree.export_graphviz(clf, out_file="mytree.dot")
  with open("mytree.dot") as f:
      dot_graph = f.read()
  graph = graphviz.Source(dot_graph)
  graph.render("mytree")
def main():

  digits = load_digits()
  X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size = 0.3, random_state = 42)
  print(X_train[1,:])
  clf = DecisionTreeClassifier(random_state=0,min_samples_leaf=5,max_leaf_nodes=10)
  #res = cross_val_score(clf, digits.data, digits.target, cv=10)
  clf.fit(X_train,y_train)
  preds = clf.predict(X_test)
  rep = metrics.classification_report(y_test,preds)
  conf_mtrx = metrics.confusion_matrix(y_test,preds)
  print(rep)
  print(conf_mtrx)
  render_tree(clf)
  #X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size = 0.3, random_state = 42)
  #id3 = ID3.ID3DecisionTreeClassifier()
  #id3.fit(X_train,X_test,)


if __name__ == "__main__": main()