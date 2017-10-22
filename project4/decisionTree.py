from __future__ import division

import math
import copy
import random
from typing import Dict, Tuple, List
from graphviz import Digraph
# import matplotlib.pyplot as plt
# import networkx as nx

from customCsvReader import CustomCSVReader


class Node:
    """
    This class represents values in the Decision tree, both leaf nodes and inner nodes.
    """
    def __init__(self, parent, decision=None, attribute_name=None, class_label=None, is_terminal=False):
        self.parent = parent
        self.parent_id = None
        self.children = []
        self.attribute = attribute_name
        self.class_label = class_label
        self.decision = decision
        self.is_terminal = is_terminal
        self.mean_value = None

        if is_terminal:
            self.attribute = None
        else:
            self.class_label = None

    def set_children(self, children):
        self.children = children

    def add_child(self, child):
        self.children.append(child)

    def get_attribute(self):
        return self.attribute

    def get_is_terminal(self):
        return self.is_terminal

    def set_parent(self, parent):
        self.parent = parent

    def get_decision(self):
        return self.decision

    def set_decision(self, decision):
        self.decision = decision

    def get_mean_value(self):
        return self.mean_value

    def set_mean_value(self, mean_value):
        self.mean_value = mean_value


class ID3:
    def __init(self):
        """
        Constructor. Init's class values.
        :return:
        """
        self.attribute_domains = {}
        self.label_index = -1
        self.possible_classes = set()

    # <editor-fold desc="Init Data">
    def create_attributes_domains(self, dataset: list) -> dict:
        """

        :param dataset:
        :return:
        """
        attributes = {}

        for datapoint in dataset:
            for attribute_index in range(len(datapoint) - 1):
                attribute = datapoint[attribute_index]
                if isinstance(attribute, float):
                    attributes[attribute_index] = "float"
                else:
                    if attribute_index not in attributes.keys():
                        attributes[attribute_index] = set()

                    attributes[attribute_index].add(attribute)

        return attributes

    def create_possible_class_list(self, dataset: list) -> set:
        """

        :param dataset:
        :return:
        """
        classes = set()

        for datapoint in dataset:
            classes.add(datapoint[-1])

        return classes

    def create_attribute(self, attribute_name, domain):
        """

        :param attribute_name:
        :param domain:
        :return:
        """
        return {
            'attribute_name': attribute_name,
            'domain': domain
        }
    # </editor-fold>

    # <editor-fold desc="ID3">
    def learn(self, training_data: List[list]):
        """

        :param training_data:
        :return:
        """
        attributes = self.create_attributes_domains(training_data)
        self.possible_classes = self.create_possible_class_list(training_data)

        return self.id3(training_data, attributes, "")

    def id3(self, data: List[list], attributes: Dict, default: str, parent: Node=None):
        """

        :param data:
        :param attributes:
        :param default:
        :param parent:
        :return:
        """
        if len(data) == 0:
            return Node(parent, class_label=default, is_terminal=True)

        if self.is_homogeneous(data):
            return Node(parent, class_label=data[0][-1], is_terminal=True)

        if len(attributes) == 0:
            return Node(parent, class_label=self.majority_label(data), is_terminal=True)

        best_attr_name = self.pick_best_attribute(data, attributes)

        node = Node(parent, attribute_name=best_attr_name)
        default_label = self.majority_label(data)

        if attributes[best_attr_name] == "float":
            first_half_subset, second_half_subset, mean_value = self.split_data_on_mean(data, best_attr_name)
            node.set_mean_value(mean_value)

            first_half_subset = copy.deepcopy(first_half_subset)
            attributes_copy = copy.deepcopy(attributes)
            del attributes_copy[best_attr_name]
            child = self.id3(first_half_subset, attributes_copy, default_label, node)
            child.set_decision("<=")
            node.add_child(child)

            second_half_subset = copy.deepcopy(second_half_subset)
            attributes_copy = copy.deepcopy(attributes)
            del attributes_copy[best_attr_name]
            child = self.id3(second_half_subset, attributes_copy, default_label, node)
            child.set_decision(">")
            node.add_child(child)

        else:
            for domain_value in attributes[best_attr_name]:
                subset = self.get_data_for_domain(data, best_attr_name, domain_value)
                subset = copy.deepcopy(subset)
                attributes_copy = copy.deepcopy(attributes)
                del attributes_copy[best_attr_name]
                child = self.id3(subset, attributes_copy, default_label, node)
                child.set_decision(domain_value)
                node.add_child(child)

        return node

    def pick_best_attribute(self, data: List[list], attributes: dict):
        """

        :param data:
        :param attributes:
        :return:
        """
        information_gained = {}
        for attribute_name, domain_list in attributes.items():

            if domain_list == "float":
                information_gain = self.get_information_gain_float_domain(data, attribute_name)
            else:
                information_gain = self.get_information_gain(data, attribute_name, domain_list)

            information_gained[attribute_name] = information_gain

        max_attribute_tuple = max(information_gained.items(), key=lambda x: x[1])

        return max_attribute_tuple[0]

    def is_homogeneous(self, records: List[list]) -> bool:
        """

        :param records:
        :return:
        """
        current_label = None
        for record in records:
            if current_label is None:
                current_label = record[-1]
            else:
                if record[-1] != current_label:
                    return False
        return True

    def majority_label(self, records: list) -> str:
        """

        :param records:
        :return:
        """

        num_label = {}

        for record in records:
            label = record[-1]
            if label not in num_label.keys():
                num_label[label] = 1
            else:
                num_label[label] += 1

        return max(num_label, key=num_label.get)

    # </editor-fold>

    # <editor-fold desc="Helpers">
    def get_label_count(self, records, label):
        count = 0
        for record in records:
            if record[-1] == label:
                count += 1
        return count

    def get_data_for_domain(self, records, attribute_name, domain):
        record_list = []
        for record in records:
            if record[attribute_name] == domain:
                record_list.append(record)

        return record_list

    # </editor-fold>

    # <editor-fold desc="Information Gain">
    def get_information_gain(self, data, attribute_name, domain_list):
        domain_entropy = []
        for domain in domain_list:
            domain_data = self.get_data_for_domain(data, attribute_name, domain)
            if len(domain_data) > 0:
                e_a = self.total_entropy(self.possible_classes, domain_data)
                domain_with_proportion = {'domain_data_count': len(domain_data), 'domain_entropy': e_a}
                domain_entropy.append(domain_with_proportion)

        weighted_entropy = self.calculate_weighted_entropy(domain_entropy, len(data))

        clazz_entropy = []
        for clazz in self.possible_classes:
            clazz_entropy.append(self.get_label_count(data, clazz))

        curr_entropy = self.sum_entropy(clazz_entropy, len(data))

        info_gain = curr_entropy - weighted_entropy

        return info_gain

    def get_information_gain_float_domain(self, data, attribute_name):
        # domains = attribute['domain']
        domain_entropy = []

        # domain_data = self.get_data_for_domain(data, attribute_name, domain)
        split_data = self.split_data_on_mean(data, attribute_name)
        split_data = list(split_data)
        mean_value = split_data[-1]
        del split_data[-1]
        # find mean
        # split data
        # for each half of the split
        #   Calculate entropy

        for data_split_part in split_data:
            if len(data_split_part) > 0:
                e_a = self.total_entropy(self.possible_classes, data_split_part)
                domain_with_proportion = {'domain_data_count': len(data_split_part), 'domain_entropy': e_a}
                domain_entropy.append(domain_with_proportion)

        weighted_entropy = self.calculate_weighted_entropy(domain_entropy, len(data))

        clazz_entropy = []
        for clazz in self.possible_classes:
            clazz_entropy.append(self.get_label_count(data, clazz))

        curr_entropy = self.sum_entropy(clazz_entropy, len(data))

        info_gain = curr_entropy - weighted_entropy

        return info_gain

    def split_data_on_mean(self, data: List[list], attribute_name) -> Tuple[list, list, float]:
        mean_attribute_value = 0
        for datapoint in data:
            mean_attribute_value += datapoint[attribute_name]
        mean_attribute_value /= len(data)

        first_half = []
        second_half = []
        for datapoint in data:
            if datapoint[attribute_name] <= mean_attribute_value:
                first_half.append(datapoint)
            else:
                second_half.append(datapoint)

        return first_half, second_half, mean_attribute_value


    def calculate_weighted_entropy(self, domain_entropies, total_data_length):
        """

        :param domain_entropies:
        :param total_data_length:
        :return:
        """
        weighted_entropy = 0
        for domain_info in domain_entropies:
            weighted_sum = domain_info['domain_data_count']/total_data_length * domain_info['domain_entropy']
            weighted_entropy += weighted_sum
        return weighted_entropy

    def total_entropy(self, list_of_classes, domain_data) -> float:
        """

        :param list_of_classes:
        :param domain_data:
        :return:
        """
        entropy_sum = 0.0
        for clazz in list_of_classes:
            count = self.get_label_count(domain_data, clazz)
            single_entropy = self.entropy(count, len(domain_data))
            entropy_sum += single_entropy

        return entropy_sum

    def sum_entropy(self, counts: list, total_length: int) -> float:
        """

        :param counts:
        :param total_length:
        :return:
        """
        entropy_sum = 0.0
        for count in counts:
            entropy_sum += self.entropy(count, total_length)

        return entropy_sum

    def entropy(self, x: float, size: int):
        """

        :param x:
        :param size:
        :return:
        """
        entropy_calculated = 0
        if x/size > 0:
            entropy_calculated = -1 * x/size * math.log(x/size, 2)
            
        return entropy_calculated

    # </editor-fold>

    # <editor-fold desc="Classify">
    def classify(self, tree: Node, test_data: List[list]):
        """

        :param tree:
        :param test_data:
        :return:
        """
        classifications = []
        if type(test_data) is not list:
            classification = self.get_classification(tree, test_data)
            classifications.append(classification)
            return classifications

        for record in test_data:
            classification = self.get_classification(tree, record)
            classifications.append(classification)

        return classifications

    def get_classification(self, node: Node, record: list):
        """

        :param node:
        :param record:
        :return:
        """
        if node.get_is_terminal():
            return node.class_label
        else:
            if node.get_mean_value() is None:
                value = record[node.attribute]
                next_nodes = [x.get_decision() for x in node.children]

                if value in next_nodes:
                    next_node_index = next_nodes.index(value)
                    next_node = node.children[next_node_index]
                    return self.get_classification(next_node, record)
            else:
                record_value = record[node.attribute]
                node_mean_value = node.get_mean_value()

                if record_value <= node_mean_value:
                    next_nodes = [n for n in node.children if n.get_decision() == "<="]
                    next_node = next_nodes[0]
                else:
                    next_nodes = [n for n in node.children if n.get_decision() == ">"]
                    next_node = next_nodes[0]

                return self.get_classification(next_node, record)
        return "ERROR"

    def evaluate(self, test_data, classifications):
        number_of_errors = 0
        for record, classification in zip(test_data, classifications):
            if record[-1] != classification:
                number_of_errors += 1

        return number_of_errors / len(test_data)
    # </editor-fold>

    # <editor-fold desc="Helper">
    def node_count(self, tree: Node):
        count = 0
        current_node = tree
        frontier = [current_node]

        while frontier:
            current_node = frontier.pop(0)
            count += 1
            frontier += current_node.children

        return count

    # </editor-fold>

    # <editor-fold desc="Draw">
    def add_nodes_to_graph(self, tree: Node, graph: Digraph) -> Digraph:
        """

        :param tree:
        :param graph:
        :return:
        """
        current_node = tree
        frontier = [current_node]
        explored = []
        node_id = 0

        while frontier:
            node_id += 1
            current_node = frontier.pop(0)
            if isinstance(current_node, Node):
                if current_node.parent is None:
                    # graph.add_node(node_id, title=current_node.attribute)
                    graph.node(str(node_id), str(current_node.attribute))
                else:
                    # graph.add_node(node_id, title=current_node.attribute)
                    # graph.add_edge(current_node.parent_id, node_id, title=current_node.get_decision())
                    graph.edge(str(current_node.parent_id), str(node_id), label=current_node.get_decision())
                    if not current_node.is_terminal:
                        graph.node(str(node_id), str(current_node.attribute))
                    elif current_node.mean_value is not None and current_node.class_label is None:
                        graph.node(str(node_id), str(current_node.get_mean_value()))
                    else:
                        graph.node(str(node_id), str(current_node.class_label))

                explored.append(current_node)
                children_to_add = copy.deepcopy(current_node.children)
                for node in children_to_add:
                    node.parent_id = node_id
                frontier += children_to_add

        return graph

    def view(self, tree):
        dot = Digraph()
        dot = self.add_nodes_to_graph(tree, dot)
        return dot

    # </editor-fold>


# reader = CustomCSVReader()
# # data = reader.read_file('data/car.data.txt', str)
# # data = reader.read_file('data/abalone.data.txt', float)
# data = reader.read_file('data/segmentation.data.new.txt', float)
# # # path = 'data/agaricus-lepiota.data'
# # # data = reader.read_file('data/agaricus-lepiota.data', str)
# #
# random.shuffle(data)
#
# half_way = int(math.floor(len(data)/3)) * 2
# set_1 = data[:half_way]
# set_2 = data[half_way:]
#
# id3 = ID3()
#
# tree_1 = id3.learn(set_1)
# # print(tree_1)
#
# dot = id3.view(tree_1)
# dot.render('test1', view=True)
#
# # evaluate
# c2 = id3.classify(tree_1, set_2)
# evaluation = id3.evaluate(set_2, c2)
# print("Error Rate = {}".format(evaluation))


# Test Part
# id3 = ID3()
# test_data = [[1],[2],[3],[4],[5],[6]]
# split_data = id3.split_data_on_mean(test_data, 0)
# split_data = list(split_data)
# del split_data[-1]
# # print(split_data[0])
# # print(split_data[1])
# # print(split_data[2])
# print(split_data)
