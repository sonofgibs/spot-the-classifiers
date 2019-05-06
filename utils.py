import random
import math

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
            convert_to_numeric(values)
            table.append(values)  # adds to the end
    infile.close()


'''
    FROM CLASS
    Converts string values to usable numeric values
    Parameter values: a list of values to be converted
    '''
def convert_to_numeric(values):
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
    add_to = 0 # Index to add to (% k) to ensure roughly equal length folds
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

# compare with 5 nearest neighbors (k=5)
def compute_class_knn(instance, vals):
    distances_and_label = [[compute_distance(val[:-1], instance[:-1]), val[-1]] for val in vals]
    distances_and_label.sort(key=lambda x: x[0]) # Sort by distance
    top_5 = [x[1] for x in distances_and_label[1:6]] # Exclude first value (test instance itself)
    predicted_class = max(set(top_5), key=top_5.count)
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