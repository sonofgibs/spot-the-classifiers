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
        return_table.append(int(float(table[index])))
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


def create_scatter_plot(yaxis_column, xaxis_column, filename, titlename, yaxis_label):
    plt.figure()
    xvalues = convert_to_numeric2(xaxis_column)
    yvalues = convert_to_numeric2(yaxis_column)
    plt.plot(xvalues, yvalues, "b.")
    # b = get_coefficient(np.array(xvalues), np.array(yvalues))
    # # predicted response vector
    # # b = b[0], m = b[1]
    # yprediction = b[0] + b[1]*np.array(xvalues)
    # # plotting the regression line
    # plt.plot(xvalues, yprediction, color="r")

    # # find correlation coefficient and covariance
    # correlation_coefficient = np.corrcoef(xvalues, yvalues)[0, 1]
    # covariance = np.cov(xvalues, yvalues)[0, 1]

    # # add annotation
    # ax = plt.gca()
    # ax.annotate("corr=%.2f, cov=%.2f" % (correlation_coefficient, covariance), xy=(
    #     .60, .95), xycoords='axes fraction', color="r", bbox=dict(boxstyle="round", fc="1", color="r"))
    plt.title(titlename + " vs Popularity")
    plt.grid(True)
    plt.ylabel(yaxis_label)
    plt.xlabel(titlename)
    plt.savefig(filename)
    plt.close()


def main():
    audio_data = []
    utils.read_file_to_table("audio_data.csv", audio_data)
    print("done")
    create_scatter_plot(utils.get_column(audio_data, 0), utils.get_column(
        audio_data, 13), "scatter_plot.pdf", "Popularity", "Acousticness")


if __name__ == "__main__":
    main()
