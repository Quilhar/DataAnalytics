import numpy as np
import pandas as pd
setx = [3, 5, 6, 8, 12, 2]

def stats(dataset):
    data = []
    data.append(dataset)
    
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

    '''Returning data_summary data frame'''
    return data_summary

final_stats = stats(setx)

final_stats

