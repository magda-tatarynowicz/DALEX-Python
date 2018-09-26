# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy as np
from sklearn import datasets, linear_model

def create_sample_linear_regression():
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only five features
    diabetes_X = diabetes.data[:,:5]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    print(diabetes_X_train)
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    return (diabetes_X_train, diabetes_y_train, regr, ['col1', 'col2', 'col3', 'dcol4','col5'])