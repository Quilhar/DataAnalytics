import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main_stats(dataset):
    '''
    Function to calculate main statistics of a dataset:
    - dataset (list or array): A list or array of numerical values
    - Returns a dictionary with mean, median, variance, std_dev, max, min
    '''
    
    data = dataset
    
    # Finding the values of the statistics
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    std_dev = np.std(data)
    max = np.max(data)
    min = np.min(data)
    
    # Converting numpy types to python types
    mean = float(mean)
    median = float(median)
    variance = float(variance)
    std_dev = float(std_dev)
    max = float(max)
    min = float(min)
    
    # Rounding to 2 decimal places
    mean = round(mean, 2)
    median = round(median, 2)
    variance = round(variance, 2)
    std_dev = round(std_dev, 2)
    max = round(max, 2)
    min = round(min, 2)
    
    # Creating a dictionary to store the results
    data_summary = {"mean": mean, "median": median, "variance": variance, "std_dev": std_dev, "max": max, "min": min}

    return data_summary

def quartiles(dataset):
    '''
    Function to calculate quartiles of a dataset:
    - dataset (list or array): A list or array of numerical values
    - Returns a dictionary with Q1, Q2 (median), and Q3
    '''
    data = dataset
    
    # Finding the quartiles
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)
    Q3 = np.percentile(data, 75)
    
    # Converting numpy types to python types
    Q1 = float(Q1)
    Q2 = float(Q2)
    Q3 = float(Q3)
    
    # Rounding to 2 decimal places
    Q1 = round(Q1, 2)
    Q2 = round(Q2, 2)
    Q3 = round(Q3, 2)
    
    # Creating a dictionary to store the results
    quartile_summary = {"Q1": Q1, "Q2": Q2, "Q3": Q3}
    
    return quartile_summary

def regular_plot(dataset, xaxis_label = '', yaxis_label = '', title = '', color = ''):
    '''
    Function to create a regular plot of a dataset:
    - dataset (list or array): A list or array of numerical values
    - xaxis_label (str): Label for the x-axis
    - yaxis_label (str): Label for the y-axis
    - title (str): Title of the plot
    - color (str): Color of the line
    '''
    
    data = dataset
    
    # Creating regular plot
    plt.plot(data, color, linestyle='-', marker='o')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title)
    
    plt.tight_layout()

def histogram(dataset, xaxis_label = '', yaxis_label = '', title = '', color = '', edgecolor = ''):
    '''
    Function to create a histogram of a dataset:
    - dataset (list or array): A list or array of numerical values
    - xaxis_label (str): Label for the x-axis
    - yaxis_label (str): Label for the y-axis
    - title (str): Title of the histogram
    - color (str): Color of the bars
    - edgecolor (str): Color of the edges of the bars
    '''
    
    data = dataset
    
    # Creating histogram
    
    # Use numpy to calculate the edges of the bins, using the 'auto' method, which calculates the optimal number of bins based on using the Freedman-Diaconis, and Sturges methods
    bins_edges_auto = np.histogram_bin_edges(data, bins='auto')
    
    plt.hist(data, bins=bins_edges_auto, edgecolor = edgecolor, color = color)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title)
    
    plt.tight_layout()
    
def piechart(dataset, labels, title = '', colors = '', explode = '', startangle = '', autopct = 'autopct'):
    '''
    Function to create a pie chart of a dataset:
    - dataset (list or array): A list or array of numerical values
    - labels (list): A list of labels for each slice
    - title (str): Title of the pie chart
    - colors (list): A list of colors for each slice
    - explode (list): A list of values to offset each slice
    - startangle (int): Starting angle of the pie chart
    - autopct (str): Format string for the percentage labels
    '''
    data = dataset
    
    # Creating pie chart
    plt.pie(data, labels=labels, colors=colors, explode=explode, startangle=startangle, autopct=autopct)
    plt.title(title)
    
    plt.tight_layout()

def boxplot(dataset, xaxis_label = '', yaxis_label = '', title = '', vert = True, color = 'white'):
    '''
    Function to create a box plot of a dataset:
    - dataset (list or array): A list or array of numerical values
    - xaxis_label (str): Label for the x-axis
    - yaxis_label (str): Label for the y-axis
    - title (str): Title of the box plot
    - vert (bool): Orientation of the box plot (True for vertical, False for horizontal)
    - color (str): Color of the box
    '''
    data = dataset
    
    # Creating box plot
    plt.boxplot(data, vert = vert, patch_artist = True, boxprops = dict(facecolor = color))
    plt.ylabel(yaxis_label)
    plt.xlabel(xaxis_label)
    plt.title(title)
    
    plt.tight_layout()

############################################################################################## Lecture 3 Functions ###########################################################################################

def find_data_correlation(x_list, y_list):
    ''' Function to find correlation coefficient between two lists of data points
    Parameters
        x_list : list
            List of data points for X
        y_list : list
            List of data points for Y
    Returns
        corelation_coefficient : float'''
    
    n = len(x_list)
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    
    sum_x = sum(x_list)
    sum_y = sum(y_list)
    sum_xy = np.sum(x_array * y_array)
    sum_x_squared = sum(x ** 2 for x in x_list)
    sum_y_squared = sum(y ** 2 for y in y_list)

    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = ((np.sqrt((n * sum_x_squared) - sum_x ** 2)) * (np.sqrt((n * sum_y_squared) - sum_y ** 2)))
    
    corelation_coefficient = numerator / denominator
            
    return corelation_coefficient
    
def least_squares_matrixes(x_data, y_data):
    ''' Function to find the least squares matrix
        x_data : list
            List of data points for X
        y_data : list
            List of data points for Y
        Returns :
            least_squares_matrix_left : matrix
                Left hand side least squares matrix
            least_squares_matrix_right : matrix
                Right hand side least squares matrix '''
         
    n = len(x_data)
    
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    
    a = sum(x_array ** 2)
    b = sum(x_array)
    c = b
    d = n
    e = sum(x_array * y_array)
    f = sum(y_array)
    
    least_squares_matrix_left = [[a, b], [c, d]]
    least_squares_matrix_right = [[e], [f]]
    
    return least_squares_matrix_left, least_squares_matrix_right

def matrix_inverse(A, ifit = 1):
    ''' Function computes the inverse matrix for either the 2X2 linear least squares matrix
    or the 3X3 quadratic least square fit, depending on the value of ifit
    
    Inputs :
        A - The least squares matrix to be inverted input as a 2 dimensional Python list
        ifit - the degree of the polynomial to be fitted (either 1 or 2)
        
    Outputs :
        A_inv - The inverse of the least squares matrix output as a 2
        dimensional Python list'''
    
    if ifit == 1:
        a_matrix = A
        
        a1 = a_matrix[0][0]
        b1 = a_matrix[0][1]
        c1 = a_matrix[1][0]
        d1 = a_matrix[1][1]
        
        det = (a1 * d1) - (b1 * c1)
        
        a2 = a_matrix[1][1]/det
        b2= -a_matrix[0][1]/det
        c2 = -a_matrix[1][0]/det
        d2 = a_matrix[0][0]/det
        
        a_inverse = [[a2, b2],
                    [c2, d2]]
        
        return a_inverse

def least_squares_coefficient(x_list, y_list):
    ''' Function to find least squares coefficients for linear fit
        A : matrix
            Left hand side least squares matrix
        B : matrix
            Right hand side least squares matrix '''

    A, B = least_squares_matrixes(x_list, y_list)
    
    A = matrix_inverse(A, 1)
    
    a = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    b = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    
    least_squares_coefficients = [[a, b]]
    
    return least_squares_coefficients

def graph__least_squares(x_data, y_data, x_axis = '', y_axix = '', title = '', fit = 1):
    ''' Function to graph least squares fit along with data points
        x_data : list
            List of data points for X
        y_data : list
            List of data points for Y
        x_axis : str
            Label for x axis
        y_axis : str
            Label for y axis
        title : str
            Title for the graph
        fit : int
            Degree of polynomial fit (1 for linear, 2 for quadratic)'''
    
    coefficients = least_squares_coefficient(x_data, y_data)
    
    plt.scatter(x_data, y_data, color='blue', label='Data Points')
    
    if fit == 1:
        a = coefficients[0][0]
        b = coefficients[0][1]
        
        y_values = [a * x + b for x in x_data]
        plt.plot(x_data, y_values, color='red', label='Least Squares Fit')
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axix)
    plt.title(title)
    plt.legend()
    plt.show()