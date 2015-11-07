import numpy as np
import math


# Constants
NASDAQ_00 = "nasdaq00.txt"
NASDAQ_01 = "nasdaq01.txt"

def get_stock_data(input_file):
    """
    Get stock data from the input file.
    """
    with open(input_file, "r") as fp:
        data = list(map(float, fp.read().split("\n")))
        return data

def get_predicted_stock_price(xi, theta):
    """
    Predict the stock price on a given day.
    """
    return (xi * theta).sum()

def get_linear_cofficients():
    """
    Get the linear model coefficients of the linear regression model.
    """
    stock_2000 = get_stock_data(NASDAQ_00)
    X_2000 = np.array([np.array([stock_2000[i-j-1] for j in range(3)]) for i in range(3, len(stock_2000))])
    Y_2000 = np.array(stock_2000[3:])
    return np.linalg.lstsq(X_2000, Y_2000)[0]

def get_mean_square_error():
    """
    Get the mean square error for stock data for years 2000 and 2001.
    """
    theta = get_linear_cofficients()
    stock_data = get_stock_data(NASDAQ_01)
    #stock_data.extend(get_stock_data(NASDAQ_01))
    X = np.array([np.array([stock_data[i-j-1] for j in range(3)]) for i in range(3, len(stock_data))])
    Y = np.array(stock_data[3:])
    return np.mean([math.pow(Y[i] - get_predicted_stock_price(X[i], theta), 2) for i in range(len(Y))])


print(get_linear_cofficients())
print(get_mean_square_error())
