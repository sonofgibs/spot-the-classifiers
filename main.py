import matplotlib.pyplot as plt
import numpy as np
import utils
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

'''
Converts string values to usable numeric values
Parameter table: a single column table to be converted
Returns return_table: a table of converted values
'''


def convert_to_numeric2(table):
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


def create_scatter_plot(xaxis_column, yaxis_column, filename, xlabel):
    plt.figure()
    xvalues = convert_to_numeric2(xaxis_column)
    yvalues = convert_to_numeric2(yaxis_column)
    plt.scatter(xvalues, yvalues, c="b")
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel("Popularity")
    plt.title(xlabel + " vs Popularity")
    plt.savefig(filename)
    plt.close()


def main():
    '''audio_data = []
    utils.read_file_to_table("audio_data.csv", audio_data)
    print("done reading")
    headers = ["Acousticness", "Danceability", "Duration", "Energy", "Instrumentalness", "Key",
               "Liveness", "Loudness", "Mode", "Speechiness", "Tempo", "Time Signature", "Valence"]
    for i in range(0, 13):
        filename = headers[i] + ".pdf"
        create_scatter_plot(utils.get_column(audio_data, i), utils.get_column(
            audio_data, 13), filename, headers[i])'''

    # kNN classifier to predict popularity (index 13)
    # using: acousticness (0), danceability (1), energy (3), instrumentalness (4),
    # liveness (6), speechiness (9), valence (12)
    # TODO: normalize and include duration, tempo, loudness
    '''trimmed_data = []
    utils.read_file_to_table("small_audio_data.csv", trimmed_data, [
                             0, 1, 3, 4, 6, 9, 12, 13])
    # generate 10 stratified cross folds
    folds = utils.stratified_cross_folds(trimmed_data, 10)
    num_correct = 0
    for i in range(0, 10):
        train, test = utils.set_up_train_test(i, folds)
        actual_popularities = [utils.discretize_popularity(x[-1]) for x in test]
        predicted_popularities = utils.knn_classifier(train, test)
        for i in range(0, len(test)):
            if actual_popularities[i] == utils.discretize_popularity(predicted_popularities[i]):
                num_correct += 1
        print(num_correct)
    accuracy = num_correct / len(trimmed_data)
    print("Accuracy: " + str(accuracy * 100))'''

    # compare with scikit-learn kNN  
    df = pd.read_csv("small_audio_data.csv")
    X = np.array(df.ix[:, 0:12]) # features
    y = np.array(df.ix[:, 13]) # class label (popularity)

    # discretize popularity
    y = [utils.discretize_popularity(val) for val in y]

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    print("Accuracy: " + str(round(accuracy_score(y_test, prediction) * 100, 2)) + "%")

if __name__ == "__main__":
    main()
