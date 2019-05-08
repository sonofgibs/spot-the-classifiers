import utils
import math
import os
#from pygraphviz import *

'''
Creates a decision tree using TDIDT algorithm
Parameter instances: table with every instance
Parameter att_indexes: list of the index of each attribute used to guess
Parameter att_domains: list of all possible answers for each attribute (e.g. ["yes","no"])
Parameter class_index: index of the attribute value that's being guessed
Parameter header: the header
Return return_tree is the decision tree
'''


def tdidt(instances, att_indexes, att_domains, class_index, header=None):
    # Basic Approach (uses recursion!):
    return_tree = []  # create the list
    # At each step, pick an attribute ("attribute selection")
    att_index = select_attribute(instances, att_indexes, class_index)
    # can't choose the same attribute twice in a branch

    # append attribute to the list
    return_tree.append("Attribute")
    return_tree.append(header[att_index])

    # remember python is pass by object reference !!!!!!
    att_indexes.remove(att_index)
    # Partition data by attribute values ... this creates pairwise disjoint partitions
    partition = partition_instances(
        instances, att_index, att_domains[att_index])
    value_index = 0  # to be able to back track later
    for value in partition:  # check each value in the partition
        value_tree = ["Value", value]

        # Repeat until one of the following occurs (base cases):

        # CASE ONE: Partition has only class labels that are the same ... no clashes, make a leaf node
        # call a function that returns true if a partition has all the same class label
        # has_same_class_label()
        is_leaf, leaf_value = check_all_same_class(
            partition[value], class_index)  # instances is the table
        if is_leaf:
            # then make leaf
            leaf_list = ["Leaves"]
            class_label_stats = compute_leaf_stats(
                instances, class_index, leaf_value)
            leaf_list.append(class_label_stats)
            value_tree.append(leaf_list)

        elif len(partition[value]) <= 1:
            # CASE THREE: No more instances to partition ... see options below
            # "backtrack" to replace attribute node with leaf node
            # can make use of compute_partition_stats() to find the class
            # of the leaf node (majority voting)

            # first, remove the attribute node
            return_tree.clear()
            return_tree = ["Leaves"]
            # majority vote for every value in partition
            max_values = []
            for value in partition:
                if partition[value]:
                    class_label_stats = compute_partition_stats(
                        instances, class_index)
                    max_value = class_label_stats[0][1]
                    max_index = 0
                    index = 0
                    for label in class_label_stats:
                        if label[1] > max_value:
                            max_value = label[1]
                            max_index = index
                        index += 1
                    max_values.append(class_label_stats[max_index])
            max_value = max_values[0][1]
            max_index = 0
            index = 0
            for i in max_values:
                if i[1] > max_value:
                    max_value = i[1]
                    max_index = index
                index += 1
            return_tree.append(max_values[max_index])
            break  # attribute is now a leaf

        elif len(att_indexes) == 0:
            # CASE TWO: No more attributes to partition ... reached the end of a branch and there may be clashes,
            # see options below
            # if we are here, then case one's boolean condition failed
            # then no more attributes
            # we have a mix class labels but no more attributes to split on
            # handle clash with majority voting

            if partition[value]:
                class_label_stats = compute_partition_stats(
                    instances, class_index)
                max_value = class_label_stats[0][1]
                max_index = 0
                index = 0
                for label in class_label_stats:
                    if label[1] > max_value:
                        max_value = label[1]
                        max_index = index
                    index += 1
                leaf_list = ["Leaves"]
                leaf_list.append(class_label_stats[max_index])
                value_tree.append(leaf_list)

            # will give list of each class label's stats for a partition
        else:
            # if none of these cases evaluate to true, then recurse
            # get new att_indexes
            new_att_indexes = att_indexes[:]
            # get new instances
            new_instances = partition[value]
            new_tree = tdidt(new_instances, new_att_indexes,
                             att_domains, class_index, header)
            value_tree.append(new_tree)
        value_index += 1
        return_tree.append(value_tree)

    return return_tree


'''
Selects an attribute to partition on based on Information gain
Parameter instances: all available instances
Parameter att_indexes: indexes of available attributes to partition on
Parameter class_index: the index of attribute being guessed
Returns att_indexes[return_index]: the index of the attribute with the largest information gain
'''


def select_attribute(instances, att_indexes, class_index):
    entropies = []
    total_instances = len(instances)

    # calculate Estart
    start_values, start_counts = utils.get_frequencies(instances, class_index)

    Estart = calc_attribute(start_counts[0], sum(start_counts))
    for aindex in att_indexes:  # find information gain of each attribute
        att_values, att_counts = utils.get_frequencies(
            instances, att_indexes[att_indexes.index(aindex)])
        Enew = 0
        for value in att_values:
            total_att_with_class = 0
            first_class = instances[0][class_index]
            # find total number of instances with the same class label and certain attribute label
            for instance in instances:
                if instance[aindex] == value:
                    if instance[class_index] == first_class:
                        total_att_with_class += 1
            att_index = att_values.index(value)
            total_count_att = att_counts[att_index]
            e_attribute = calc_attribute(total_att_with_class, total_count_att)
            Enew += (total_count_att/total_instances) * e_attribute
        Info_gain = Estart - Enew
        entropies.append(Info_gain)
    if entropies:
        return_index = entropies.index(max(entropies))
    else:
        return_index = 0
    return att_indexes[return_index]


'''
Helper function for select_attribute()
Parameter total_att_with_class: total number of instances with a certain attribute label and same class label
Parameter total_count_att: total number of instances with the same certain attribute label
Returns e_attribute: the partition entropy
'''


def calc_attribute(total_att_with_class, total_count_att):
    p_1 = total_att_with_class / total_count_att
    p_2 = (total_count_att -
           total_att_with_class) / total_count_att
    # can't include zero values
    if p_1 == 0:
        e_attribute = -(p_2 * math.log(p_2, 2))
    elif p_2 == 0:
        e_attribute = -(p_1 * math.log(p_1, 2))
    else:
        e_attribute = -(p_1 * math.log(p_1, 2)) - \
            (p_2 * math.log(p_2, 2))
    return e_attribute


'''
Partitions a dataset based on an attribute
'''


def partition_instances(instances, att_index, att_domain):
    # this is a group by attribute domain, not by att_values in the table (instances)
    partition = {}
    for att_value in att_domain:
        subinstances = []
        cutoff, start_value = get_cutoff_value(att_value)
        for instance in instances:
            # check if this instance has att_value at att_index
            if instance[att_index] <= cutoff and instance[att_index] >= start_value:
                subinstances.append(instance)
        partition[att_value] = subinstances
    return partition


'''
Finds the value which each branch will be cutoff on
Returns: cutoff value and the value which the range starts on
'''


def get_cutoff_value(att_value):
    if att_value == ">=0.25":
        # -1 because this range should be inclusive of 0
        return 0.25, -1
    elif att_value == ">=0.50":
        return 0.50, 0.25
    elif att_value == ">=0.75":
        return 0.75, 0.50
    else:
        return 1.0, 0.75


'''
Helpful for base case #1 (all class labels are the same... make a leaf node)
Returns True if all instances have same label, else False
Returns leaf_value: the class_label value of the leaf node(if true)
'''


def check_all_same_class(value, class_index):
    leaf_value = None
    # True if all instances have same label
    index = 0
    if value:
        # if there is a value
        first_instance = value[0]
        leaf_value = first_instance[class_index]
        for instance in value:
            if instance[class_index] != first_instance[class_index]:
                return False, None
    else:
        return False, None
    return True, leaf_value


'''
In order to handle leaves with clashes for case 2
'''


def compute_leaf_stats(instances, class_index, leaf_value):
    return_list = []
    # ["True", 3, 5, 60%]
    # index 2 is all instances in attribute
    # index 1 is how many have a certain class index
    # index 0 is that certain class index
    values, counts = utils.get_frequencies(instances, class_index)
    count_index = 0
    for value in values:
        if value == leaf_value:
            return_list.append(value)
            return_list.append(counts[count_index])
            return_list.append(len(instances))
            return return_list
        count_index += 1
    return return_list


'''
Helpful for base case #2 (no more attributes to partition, need to handle clashes)
Return a list of stats
'''


def compute_partition_stats(instances, class_index):
    # Return a list of stats: [[label1, occ1, tot1], [label2, occ2, tot2], ...]
    return_list = []
    # ["True", 3, 5, 60%]
    # index 2 is all instances in attribute
    # index 1 is how many have a certain class index
    # index 0 is that certain class index
    values, counts = utils.get_frequencies(instances, class_index)
    count_index = 0
    for value in values:
        value_list = []
        value_list.append(value)
        value_list.append(counts[count_index])
        value_list.append(len(instances))
        return_list.append(value_list)
        count_index += 1
    return return_list


'''
Creates a .dot file and .pdf for visual representation of tree
Parameter tree: the tree to draw
Parameter labels: the labels to place on drawing
Parameter filename: name to label files
'''


def create_dot_tree(tree, labels, filename):
    name = filename + ".dot"
    G = AGraph()
    create_tree(tree, labels, 0, G)
    G.write(name)
    os.system("dot -Tpdf -o " + filename + ".pdf " + name)


'''
Recursively creates the .dot file
Parameter tree: remaining instances to draw
Parameter node_name: number to create unique node labels
Parameter G: the file to draw to
Returns node_name: number to create unique node labels
'''


def create_tree(tree, labels, node_name, G):
    # leaf case
    if tree[0] == "Leaves":
        leaf_key = str(tree[1][0]) + str(node_name)
        G.add_node(leaf_key, label=tree[1][0], shape='box')

    # assign attribute
    if tree[0] == "Attribute":
        att_key = tree[1] + str(node_name)
        label_index = labels[tree[1]]
        G.add_node(att_key, label=label_index, shape='diamond')
    # find which value to go down
    for i in range(2, len(tree)):
        val_key = str(tree[i][1]) + str(node_name)
        # if next is leaf
        if tree[i][2][0] == "Leaves":
            next_node = tree[i][2][1][0]
        else:
            next_node = tree[i][2][1]
        next_key = str(next_node) + str(node_name)
        node_name = create_tree(
            tree[i][2], labels, node_name, G) + 1
        G.add_edge(att_key, next_key, label=tree[i][1])

    return node_name


'''
Uses the tree to predict the class label for the instance
Takes a decision tree (produced by tdidt()) and an instance to classify
Returns the predicted label for the instance
'''


def tdidt_classifier(unseen_instance, tree, header):
    # leaf case
    if tree[0] == "Leaves":
        return tree[1][0]
    # assign attribute
    attribute = tree[1]
    attribute_index = header.index(attribute)
    # find which value to go down
    for i in range(2, len(tree)):
        unseen = get_range_value(unseen_instance[attribute_index])
        if tree[i][1] == unseen:
            return tdidt_classifier(unseen_instance, tree[i][2], header)


def get_range_value(value):
    if value >= 0.25:
        return '>=0.25'
    elif value >= 0.50:
        return '>=0.50'
    elif value >= 0.75:
        return '>=0.75'
    else:
        return '>=1.0'


def tree_classifier(train, test, att_indexes, att_domains, class_index, col_names):
    predicted_classes = []
    tree = tdidt(train, att_indexes, att_domains, class_index, col_names)
    for instance in test:
        predicted_class = tdidt_classifier(instance, tree, col_names)
        predicted_classes.append(predicted_class)
    return predicted_classes
