import matplotlib.pyplot as plt
import numpy as np
import utils


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


def create_scatter_plot(yaxis_column, xaxis_column, filename, ylabel):
    plt.figure()
    xvalues = convert_to_numeric2(xaxis_column)
    yvalues = convert_to_numeric2(yaxis_column)
    plt.scatter(xvalues, yvalues, c="b")
    plt.grid(True)
    plt.ylabel(ylabel)
    plt.xlabel("Popularity")
    plt.title(ylabel + " vs Popularity")
    plt.savefig(filename)
    plt.close()


def main():
    audio_data = []
    utils.read_file_to_table("audio_data.csv", audio_data)
    print("done reading")
    headers = ["Acousticness", "Danceability", "Duration", "Energy", "Instrumentalness", "Key",
               "Liveness", "Loudness", "Mode", "Speechiness", "Tempo", "Time Signature", "Valence"]
    for i in range(0, 13):
        print(i)
        filename = headers[i] + ".pdf"
        create_scatter_plot(utils.get_column(audio_data, i), utils.get_column(
            audio_data, 13), filename, headers[i])


if __name__ == "__main__":
    main()
