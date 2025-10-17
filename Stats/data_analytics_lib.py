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