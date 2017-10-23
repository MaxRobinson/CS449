import copy
import random
from typing import Dict, Tuple, List

import math

from customCsvReader import CustomCSVReader
from decisionTree import ID3, Node

class ID3Pruning:

    def prune(self, tree, validation_set):
        """
        Main algorithm for doing Reduced error pruning
        This is a bottom up algorithm that starts at the bottom of the tree and replaces
        nodes with the majority label.
        The new tree is tested to see if the error rate has improved. If so, keep the change, if not revert.

        Psuedo code:

        change_made_to_tree = false
        while change_made_to_tree true && tree(root).class_label is None
            change_made_to_tree = false
            get leaf node parents
            for each leaf node parent
                get the majority label for that node
                replace the node with majority class label

                Test tree on validation set
                If improved accuracy (or same), keep
                    change_made_to_tree = true



        """
        best_error_rate = self.get_error_rate(tree, validation_set)

        change_made_to_tree = True
        while change_made_to_tree and tree.class_label is None:

            change_made_to_tree = False
            nodes_to_remove = self.get_leaf_parents(tree)

            for node in nodes_to_remove:
                majority_label = self.get_majority_label(node)

                node.class_label = majority_label
                node.is_terminal = True

                error_rate = self.get_error_rate(tree, validation_set)
                if error_rate <= best_error_rate:
                    best_error_rate = error_rate
                    change_made_to_tree = True
                    node.children = []
                else:
                    node.class_label = None
                    node.is_terminal = False

        return tree

    def get_leaf_parents(self, tree):
        """
        A BFS algorithm that searches to find all parents of leaf nodes.
        returns a list of nodes that are parents of leaf nodes.
        """
        current_node = tree
        frontier = [current_node]
        explored = []

        list_of_leaf_parents = []

        while frontier:

            current_node = frontier.pop(0)

            if self.is_parent_of_only_leaf_nodes(current_node):
                list_of_leaf_parents.append(current_node)
            else:
                frontier += current_node.children

            explored.append(current_node)

        return list_of_leaf_parents

    def is_parent_of_only_leaf_nodes(self, node: Node):
        """
        A helper function to check if a node is a parent of only leaf nodes.
        This is important so that we do not get intermediate nodes to try and replace.
        """
        if node.is_terminal:
            return False

        for child in node.children:
            if not child.is_terminal:
                return False
        return True

    def get_majority_label(self, node: Node) -> str:
        """
        returns the majority for a node that is a likely a parent node of a leaf
        Uses a BFS

        """
        num_label = {}

        current_node = node
        frontier = [current_node]
        explored = []

        while frontier:
            current_node = frontier.pop(0)

            if current_node.is_terminal:
                label = current_node.class_label
                if label not in num_label.keys():
                    num_label[label] = 1
                else:
                    num_label[label] += 1
            else:
                frontier += current_node.children

        return max(num_label, key=num_label.get)

    def get_error_rate(self, tree: Node, validation_set: List[list]) -> float:
        """
        A helper function that calculates the error rate of a tree.
        Uses the ID3 classificaiton and evaluation function to get the results.

        Used to determine if a tree got bettter of not.
        """
        id3 = ID3()
        classifications = id3.classify(tree, validation_set)
        error_rate = id3.evaluate(validation_set, classifications)
        # print(error_rate)
        return error_rate


""" Local Test for correctness """
# reader = CustomCSVReader()
# # data = reader.read_file('data/segmentation.data.new.txt', float)
# data = reader.read_file('data/car.data.txt', str)
# # data = reader.read_file('data/abalone.data.txt', float)
#
# random.shuffle(data)
#
# half_way = int(math.floor(len(data)/3)) * 2
# set_1 = data[:half_way]
# set_2 = data[half_way:]
#
# id3 = ID3()
#
# tree_1 = id3.learn(set_1)
# tree1_dot = id3.view(tree_1)
# tree1_dot.render('test_original', view=True)
# # print("Original Tree Node Count = {}".format(id3.node_count(tree_1)))
#
# prune = ID3Pruning()
# pruned_tree = prune.prune(copy.deepcopy(tree_1), set_2)
#
# classification_original = id3.classify(tree_1, set_2)
# classification_pruned = id3.classify(pruned_tree, set_2)
#
# error_rate_original = id3.evaluate(set_2, classification_original)
# error_rate_pruned = id3.evaluate(set_2, classification_pruned)
#
# print(error_rate_original)
# print(error_rate_pruned)

# pruned_tree_dot = id3.view(pruned_tree)
# pruned_tree_dot.render('test_pruned', view=True)
#
# print("Original Tree Node Count = {}".format(id3.node_count(tree_1)))
# print("Pruned Tree Node Count = {}".format(id3.node_count(pruned_tree)))


