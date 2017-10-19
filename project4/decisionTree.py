from __future__ import division

import math
import copy
import random

from customCsvReader import CustomCSVReader

class Node:
    def __init__(self, attribute_name):
        self.children = []
        self.attribute = attribute_name
        # self.majority_label = majority_label

    def set_children(self, children):
        self.children = children

    def add_child(self, child):
        self.children.append(child)

    def get_attribute(self):
        return self.attribute


class ID3:
    def __init(self):
        self.attribute_domains = {}
        self.label_index = -1


    # <editor-fold desc="Init Data">
    def create_attributes_domains(self, dataset: list) -> dict:
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


    # def attributes_domains(self):
    #     return {
    #         'label': ['e', 'p', '?'],
    #         'cap-shape': ['b', 'c', 'x', 'f', 'k', 's', '?'],
    #         'cap-surface': ['f', 'g', 'y', 's', '?'],
    #         'cap-color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y', '?'],
    #         'bruises?': ['t', 'f', '?'],
    #         'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's', '?'],
    #         'gill-attachment': ['a', 'd', 'f', 'n', '?'],
    #         'gill-spacing': ['c', 'w', 'd', '?'],
    #         'gill-size': ['b', 'n', '?'],
    #         'gill-color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y', '?'],
    #         'stalk-shape': ['e', 't', '?'],
    #         'salk-root': ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    #         'stalk-surface-above-ring': ['f', 'y', 'k', 's', '?'],
    #         'stalk-surface-below-ring': ['f', 'y', 'k', 's', '?'],
    #         'stalk-color-above-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y', '?'],
    #         'stalk-color-below-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y', '?'],
    #         'veil-type': ['p', 'u', '?'],
    #         'veil-color': ['n', 'o', 'w', 'y', '?'],
    #         'ring-number': ['n', 'o', 't', '?'],
    #         'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z', '?'],
    #         'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y', '?'],
    #         'population': ['a', 'c', 'n', 's', 'v', 'y', '?'],
    #         'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd', '?'],
    #     }


    def get_positive_label(self):
        return 'e'


    def get_negative_label(self):
        return 'p'


    def create_attribute(self, attribute_name, domain):
        return {
            'attribute_name': attribute_name,
            'domain': domain
        }

    def get_domain(self, attribute):
        attributes = self.attributes_domains()
        return attributes[attribute]
    # </editor-fold>

    # <editor-fold desc="ID3">
    def learn(self, training_data):
        attributes = self.create_attributes_domains(training_data)
        return self.id3(training_data, attributes, "")


    def id3(self, data, attributes, default):
        if len(data) == 0:
            return default

        if self.is_homogeneous(data):
            return data[0][-1]

        if len(attributes) == 0:
            self.majority_label(data)

        best_attr_name = self.pick_best_attribute(data, attributes)

        node = Node(best_attr_name)
        default_label = self.majority_label(data)

        for domain_value in attributes[best_attr_name]:
            subset = self.get_data_for_domain(data, best_attr_name, domain_value)
            subset = copy.deepcopy(subset)
            attributes_copy = copy.deepcopy(attributes)
            del attributes_copy[best_attr_name]
            child = self.id3(subset, attributes_copy, default_label)
            child_dict = {"decision": domain_value, "child": child, "parent": node}
            node.add_child(child_dict)

        return node

    # ToDo: Test Me!
    def pick_best_attribute(self, data: list, attributes: dict):
        information_gained = {}
        for attribute_name, domain in attributes.items():
            attribute = self.create_attribute(attribute_name, domain)
            information_gain = self.get_information_gain(data, attribute)
            information_gained[attribute_name] = information_gain

        max_attribute_tuple = max(information_gained.items(), key=lambda x: x[1])

        return max_attribute_tuple[0]

    def is_homogeneous(self, records: list) -> bool:
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
            if record['label'] == label:
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
    # ToDo: Test Me!
    def get_information_gain(self, data, attribute):
        domains = attribute['domain']
        domain_entropy = []
        for domain in domains:
            domain_data = self.get_data_for_domain(data, attribute['attribute_name'], domain)
            if len(domain_data) > 0:
                # TODO -> Generalize to n possible classes
                positive_count = self.get_label_count(domain_data, self.get_positive_label())
                negative_count = self.get_label_count(domain_data, self.get_negative_label())
                e_a = self.entropy(positive_count, negative_count, positive_count + negative_count)
                domain_with_proportion = {'domain_data_count': len(domain_data), 'domain_entropy': e_a}
                domain_entropy.append(domain_with_proportion)

        # weighted_entropy  = sum([(x['domain_data_count']/len(data) * x['domain_entropy']) for x in domain_entropy])
        weighted_entropy = self.calculate_weighted_entropy(domain_entropy, len(data))

        positive_count = self.get_label_count(data, self.get_positive_label())
        negative_count = self.get_label_count(data, self.get_negative_label())

        curr_entropy = self.entropy(positive_count, negative_count, len(data))

        info_gain = curr_entropy - weighted_entropy

        return info_gain

    def calculate_weighted_entropy(self, domain_entropies, total_data_length):
        weighted_entropy = 0
        for domain_info in domain_entropies:
            weighted_sum = domain_info['domain_data_count']/total_data_length * domain_info['domain_entropy']
            weighted_entropy += weighted_sum
        return weighted_entropy

    def entropy(self, x, y, size):
        part_1 = 0
        part_2 = 0
        if x/size > 0:
            part_1 = -1 * x/size * math.log(x/size, 2)

        if y/size > 0:
            part_2 = -1 * y/size * math.log(y/size, 2)

        return part_1 + part_2

    # </editor-fold>

    # <editor-fold desc="Classify">
    def classify(self, tree, test_data):
        classifications = []
        if type(test_data) is not list:
            classification = self.get_classification(tree, test_data)
            classifications.append(classification)
            return classifications

        for record in test_data:
            classification = self.get_classification(tree, record)
            classifications.append(classification)

        return classifications

    def get_classification(self, node, record):
        actual_node = None
        if isinstance(node, Node):
            actual_node = node
        elif isinstance(node, dict) and 'child' in node:
            actual_node = node['child']

        if isinstance(actual_node, str):
            return actual_node

        if isinstance(actual_node, Node):
            value = record[actual_node.attribute]

            next_nodes = [x['decision'] for x in actual_node.children]

            if value in next_nodes:
                next_node_index = next_nodes.index(value)
                next_node = actual_node.children[next_node_index]
                return self.get_classification(next_node, record)
        return "ERROR"
    # </editor-fold>

    # <editor-fold desc="Draw">
    def add_nodes_to_graph(self, tree, graph):
        current_node = tree
        frontier = [current_node]
        explored = []
        node_id = 0

        while frontier:
            current_node = frontier.pop(0)
            if isinstance(current_node, Node):
                graph.add_node(node_id, title=current_node.attribute)
                # graph.node(str(node_id), current_node.attribute)

                explored.append(current_node.attribute)
                children_to_add = copy.deepcopy(current_node.children)
                for node in children_to_add:
                    node['parent_id'] = node_id
                frontier += children_to_add

            if isinstance(current_node, dict):
                node_value = current_node['child']

                if isinstance(node_value, Node):
                    node_id += 1
                    graph.add_node(node_id, title=node_value.attribute)
                    graph.add_edge(current_node['parent_id'], node_id, title=current_node['decision'])
                    # graph.node(str(node_id), node_value.attribute)
                    # graph.edge(str(current_node['parent_id']), str(node_id), label=current_node['decision'])
                    explored.append(node_value.attribute)
                    children_to_add = copy.deepcopy(node_value.children)
                    for node in children_to_add:
                        node['parent_id'] = node_id
                    frontier += children_to_add
                if isinstance(node_value, str):
                    # node_id = uuid.uuid4()
                    node_id += 1
                    graph.add_node(node_id, title=node_value)
                    graph.add_edge(current_node['parent_id'], node_id, title=current_node['decision'])
                    # graph.node(str(node_id), node_value)
                    # graph.edge(str(current_node['parent_id']), str(node_id), label=current_node['decision'])
                    explored.append(node_value)

        return graph


    # def view(tree):
    #     graph = nx.DiGraph()
    #     graph = add_nodes_to_graph(tree, graph)
    #
    #     pos = nx.spring_layout(graph)
    #     nx.draw(graph, pos, arrows=False)
    #
    #     edge_labels = dict([((u,v,),d['title']) for u,v,d in graph.edges(data=True)])
    #     node_labels = dict([(u, d['title']) for u,d in graph.nodes(data=True)])
    #
    #     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    #     nx.draw_networkx_labels(graph, pos, labels=node_labels)
    #
    #     plt.show()
    #     #
    #     # dot = Digraph()
    #     # dot = add_nodes_to_graph(tree, dot)
    #     # return dot
    #
    #

    # </editor-fold>



# <editor-fold desc="Tests Old">
# test_records = read_file()
#
# random.shuffle(test_records)
#
# half_way = int(math.floor(len(test_records)/2))
# set_1 = test_records[:half_way]
# set_2 = test_records[half_way:]
#
# # build
# tree_1 = id3(set_1, attributes_domains(), 'e')
#
# # view
# # dot = view(tree_1)
# # print dot.source
# # dot.render('test1', view=True)
#
#
# # evaluate 2
# c1 = classify(tree_1, set_1)
# evaluation = evaluate(set_1, c1)
# print( "Error Rate = {}".format(evaluation))
#
# # evaluate
# c2 = classify(tree_1, set_2)
# evaluation = evaluate(set_2, c2)
# print("Error Rate = {}".format(evaluation))
#
#
#
#
#
# # build
# tree_2 = id3(set_2, attributes_domains(), 'e')
#
# # view
# # dot = view(tree_2)
# # dot.render('test2', view=True)
#
# # evaluate 2
# c1 = classify(tree_2, set_1)
# evaluation = evaluate(set_1, c1)
# print("Error Rate = {}".format(evaluation))
#
# # evaluate
# c2 = classify(tree_2, set_2)
# evaluation = evaluate(set_2, c2)
# print("Error Rate = {}".format(evaluation))

# </editor-fold>


reader = CustomCSVReader()
data = reader.read_file('data/car.data.txt', str)

id3 = ID3()

attributes = id3.create_attributes_domains(data)

print(attributes)
# tree_1 = id3(set_1, attributes_domains(), 'e')



