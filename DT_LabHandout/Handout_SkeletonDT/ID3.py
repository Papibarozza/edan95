from collections import Counter
from graphviz import Digraph
import numpy as np
import helpers

class Tree:
    def __init__(self):
        self.splitting_criterion = None
        self.label = None
        self.children = []
        
class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

        self.__tree = Tree()


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)
        
        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self):

        # Change this to make some more sense
        return None


    def buildTree(self,data,target,attributes,classes,node):

        if(len(attributes)==0):
            node.label = helpers.most_common(target)
            return node
        else:
            split_attr = helpers.get_best_split(target,classes,data,attributes)
            node.splitting_criterion = split_attr
            for val in attributes[split_attr] :
                target_idxs = [ i for i in range(len(data)) if val in data[i] ]

                #To do array indexing...
                tar = np.array(list(target))
                dat = np.array(list(data))

                #Partitions data
                partition_target = tuple(tar[target_idxs])
                partition = [tuple(elem) for elem in dat[target_idxs]]

                #If it would not result in an empty partition, or we get no new information:
                if(len(partition)!=0 and (len(partition_target) != len(target)) ):
                    node.children.append(self.buildTree(partition,partition_target,{key:attributes[key] for key in attributes if key!=split_attr},classes,Tree()))
                else:
                    leaf = Tree()
                    leaf.label = helpers.most_common(target)
                    node.children.append(leaf) 
                #connect node to children
                for child in node.children:
                    curr_node = {'id': self.__nodeCounter, 'label': child.label, 'attribute': child.splitting_criterion, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}
                    self.__nodeCounter+=1
                    self.add_node_to_graph(curr_node,self.__nodeCounter)
            
            return node
    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        
        # fill in something more sensible here... root should become the output of the recursive tree creation
        root = self.buildTree(data,target,attributes,classes, Tree())
        for child in root:
            curr_node = {'id': self.__nodeCounter, 'label': child.label, 'attribute': child.splitting_criterion, 'entropy': None, 'samples': None,
                    'classCounts': None, 'nodes': None}
            self.__nodeCounter+=1
        self.add_node_to_graph(curr_node,self.__nodeCounter)
        
        return root

        

    def predict(self, data, tree) :
        predicted = list()

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted
    
   
        

        


