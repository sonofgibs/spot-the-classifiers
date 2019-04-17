import random

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


def read_file_to_table(filename, table):
    infile = open(filename, "r")
    lines = infile.readlines()
    for line in lines:
        # get rid of newline character
        line = line.strip()
        # now we want to break line into strings, using the comma as a delimiter
        values = line.split(",")
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
            numeric_val = int(values[i])
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
