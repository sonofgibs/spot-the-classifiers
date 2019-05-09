import random
import matplotlib.pyplot as plt
import math
import numpy as np

'''
Converts string values to usable numeric values
Parameter table: a single column table to be converted
Returns return_table: a table of converted values
'''


def convert_to_numeric(table):
    return_table = []
    index = 0
    for value in table:
        return_table.append(float(table[index]))
        index += 1
    return return_table


'''
Finds the coefficient for the linear regression of a scatterplot
Parameter xvalues: the table of values for the x axis
Parameter yvalues: the table of values of the y axis
Returns (b1,b2): the slope and y intercept of the linear regression
'''


def get_coefficient(xvalues, yvalues):
    size = np.size(xvalues)
    # find the mean of xvalues and yvalues
    xmean = np.mean(xvalues)
    ymean = np.mean(yvalues)
    # calculate the cross-deviation and deviation about x
    cross_deviation = np.sum(yvalues*xvalues) - size*ymean*xmean
    xdeviation = np.sum(xvalues*xvalues) - size*xmean*xmean
    # calculate the regression coefficients
    b2 = cross_deviation / xdeviation  # this is the slope (m)
    b1 = ymean - b2*xmean  # this is y intercept (b)
    return (b1, b2)


'''
Creates a scatter plot comparing two data tables
Parameter yaxis_column: single column table to be plotted on the y axis
Parameter xaxis_column: single column table to be plotted on the x axis
Parameter filename: the name of the file to save the graph to
Parameter titlename: the title for the graph
Parameter yaxis_label: the label for the y axis
Parameter step: the step of the program to plot
'''


def create_scatter_plot(xaxis_column, yaxis_column, xlabel):
    plt.figure()
    xvalues = convert_to_numeric(xaxis_column)
    yvalues = convert_to_numeric(yaxis_column)
    plt.scatter(xvalues, yvalues, c="b")
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel("Popularity")
    plt.title(xlabel + " vs Popularity")
    plt.show()


'''
FROM CLASS
Gets the data from a single attribute
Parameter table: the full table to grab the column from
Parameter column_index: the number of column to grab
Returns column: a table of all values from a singular column
'''


def get_column(table, column_index):
    column = []
    for row in table:
        if row[column_index] != "NA":
            column.append(row[column_index])
    return column


def get_frequencies(table, column_index):
    column = sorted(get_column(table, column_index))
    values = []
    counts = []

    for value in column:
        if value not in values:
            values.append(value)
            # first time we have seen this value
            counts.append(1)
        else:  # we've seen it before, the list is sorted...
            counts[-1] += 1

    return values, counts


'''
    FROM CLASS
    Opens a file and reads information to a file
    Parameter filename: the name a file that will be open and read
    Parameter table: a table to hold the information in the file
    '''


def read_file_to_table(filename, table, indices=None):
    infile = open(filename, "r")
    lines = infile.readlines()
    for line in lines:
        # get rid of newline character
        line = line.strip()
        # now we want to break line into strings, using the comma as a delimiter
        values = line.split(",")
        if indices is not None:
            trimmed_values = []
            # only append given indices
            for i in range(0, len(values)):
                if i in indices:
                    trimmed_values.append(float(values[i]))
            table.append(trimmed_values)
        else:
            convert_to_numeric1(values)
            table.append(values)  # adds to the end
    infile.close()


'''
    FROM CLASS
    Converts string values to usable numeric values
    Parameter values: a list of values to be converted
    '''


def convert_to_numeric1(values):
    for i in range(len(values)):
        try:
            numeric_val = float(values[i])
            # success
            values[i] = numeric_val
        except ValueError:
            pass


def randomize_table(table):
    randomized = table[:]
    n = len(table)
    for i in range(n):
        j = random.randrange(0, n)
        randomized[i], randomized[j] = randomized[j], randomized[i]
    return randomized


def stratified_cross_folds(table, k):
    # Generate k folds
    folds = []
    randomized = randomize_table(table)
    partitions = {}
    for row in randomized:
        if row[-1] in partitions:
            vals = partitions.get(row[-1])
            vals.append(row)
        else:
            partitions[row[-1]] = [row]
    # Set up folds
    for i in range(0, k):
        fold = []
        folds.append(fold)
    add_to = 0  # Index to add to (% k) to ensure roughly equal length folds
    for key in partitions:
        vals = partitions.get(key)
        for i in range(0, len(vals)):
            folds[add_to % k].append(vals[i])
            add_to += 1
    return folds


def set_up_train_test(i, folds):
    test = folds[i]
    if i == len(folds) - 1:
        train = folds[:i]
    elif i == 0:
        train = folds[i+1:]
    else:
        train = folds[:i] + folds[i+1:]
    flat_train = []
    for fold in train:
        for row in fold:
            flat_train.append(row)
    return flat_train, test


def compute_distance(v1, v2):
    assert(len(v1) == len(v2))
    dist = math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

# compare with 8 nearest neighbors (k=8)


def compute_class_knn(instance, vals):
    distances_and_label = [
        [compute_distance(val[:-1], instance[:-1]), val[-1]] for val in vals]
    distances_and_label.sort(key=lambda x: x[0])  # Sort by distance
    # Exclude first value (test instance itself)
    top_8 = [x[1] for x in distances_and_label[1:9]]
    predicted_class = max(set(top_8), key=top_8.count)
    return predicted_class


def normalize(vals):
    # (x - min(xs)) / ((max(xs) - min(xs)) * 1.0)
    min_val = min(vals)
    max_val = max(vals)
    return [round((x - min_val) / ((max_val - min_val) * 1.0), 4) for x in vals]


def knn_classifier(train, test):
    predicted_classes = []
    for instance in test:
        predicted_class = compute_class_knn(instance, train)
        predicted_classes.append(predicted_class)
    return predicted_classes


def discretize_popularity(val):
    if val <= 25:
        return 1
    elif val <= 50:
        return 2
    elif val <= 75:
        return 3
    else:
        return 4

# Naive Bayes helper functions
# =========================================================


def compute_probabilities(table):
    num_rows = len(table)
    classes = get_column(table, -1)
    priors = {}
    for c in classes:
        if c in priors:
            priors[c] += 1
        else:
            priors[c] = 1
    # Divide by number of instances:
    for key in priors:
        priors[key] /= num_rows
    return priors


def naive_bayes_classifier(priors, test, table):
    classes = list(priors.keys())
    probabilities = {}
    for c in classes:
        probability = priors[c]
        for i in range(0, len(test)):
            # Calculate the Gaussian posterior for each attribute
            # Class label = last attribute (index = -1)
            mean, std = mean_std_att(
                table, index=i, class_label=c, class_index=-1)
            posterior = gaussian(test[i], mean, std)
            probability *= posterior
        probabilities[c] = probability
    predicted_class = max(probabilities.items(), key=lambda x: x[1])
    return predicted_class[0]


def gaussian(x, mean, sdev):
    first, second = 0, 0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
    return first * second


def mean_std_att(table, index, class_label, class_index):
    vals = []
    for row in table:
        if row[class_index] == class_label:
            vals.append(row[index])
    mean = np.average(vals)
    std = np.std(vals)
    return mean, std
