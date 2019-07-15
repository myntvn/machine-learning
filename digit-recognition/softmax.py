import numpy as np
import scipy.sparse as sparse

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each data point X[i], the probability that X[i] is
    labeled as j for j = 0, 1, ..., k-1

    Args:
        X: (n, d) numpy array (n data points each with d features)
        theta: (k, d) numpy array where row j represents the parameters of the model for label j
        temp_parameter: the temperature parameter of softmax function
    Returns:
        h: (k, n) numpy array where each entry [i][j] is the probability that X[i] is labeled as j
    """
    h = np.dot(theta, X.T)
    h = h / temp_parameter

    # get max value of each column
    c = h.max(axis=0)

    h = h - c

    h = np.exp(h)

    s = 1 / np.sum(h, axis=0)

    return s * h


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost for every data points

    Args:
        X: (n, d) numpy array (n data points each with d features)
        Y: (n, ) numpy array containing the labels (number from 0-9) for each data point
        theta: (k, d) numpy array where row j represents the parameters of the model for label j
        lambda_factor: the regularization constant
        temp_parameter: the temperature parameter of softmax function
    Returns:
        c: the cost value
    """
    h = compute_probabilities(X, theta, temp_parameter)

    cost = 0
    for i in range(X.shape[0]):
        for j in range(theta.shape[0]):
            if Y[i] == j:
                cost += np.log(h[j,i])

    cost = -cost / X.shape[0]

    theta = np.power(theta, 2)

    cost += lambda_factor / 2 * theta.sum()

    return cost


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X: (n, d) numpy array (n data points each with d features)
        Y: (n, ) numpy array containing the labels (number from 0-9) for each data point
        theta: (k, d) numpy array where row j represents the parameters of the model for label j
        alpha: the learning rate
        lambda_factor: the regularization constant
        temp_parameter: the temperature parameter of softmax function
    Returns:
        theta: (k, d) numpy array that is the final value of parameter theta
    """
    delta = sparse.coo_matrix(theta.shape).toarray()

    h = compute_probabilities(X, theta, temp_parameter)

    for j in range(delta.shape[0]):
        y = Y
        y = np.where(y != j, 0, 1)
        p = y - h[j]

        x = X.T * p
        x = x.T
        x = x.sum(axis=0)

        grad = -x / (temp_parameter * X.shape[0]) + lambda_factor * theta[j]

        delta[j] += grad

    theta -= alpha * delta

    return theta


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3) labels

    Args:
        train_y: 1D array containing labels of the training set
        test_y: 1D array containing labels of the test set
    Returns:
        train_y_mod3: 1D array containing new labels (0-2) of the training set
        test_y_mod3: 1D array containing new labels (0-2) of the test set
    """
    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)

    return train_y_mod3, test_y_mod3


def softmax_error_mod3(X, Y, theta, temp_parameter):
    """
    Computes the error of new labels when the classifier predicts the digit

    Args:
        X: 2D numpy array represents data points need to be classified
        Y: 1D numpy array represents the label (0-2) of data points
        theta: 2D numpy array where row j represents the parameters of the model for label j
        temp_parameter: temperature parameter of softmax function
    Returns:
        test error
    """
    pred = predict(X, theta, temp_parameter)
    pred = np.mod(pred, 3)

    return 1- np.mean(pred == Y)

def augment_feature_vector(X):
    """
    Adds a feature with value 1 at the begin for each data point

    Args:
        X: 2D numpy array represents data points
    Returns:
        X with added feature for each data point
    """
    column_of_ones = np.zeros([len(X), 1]) + 1

    return np.hstack((column_of_ones, X))


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset with theta
    initialized to the all-zeros array.

    Args:
        X: (n, d) numpy array (n data points each with d features)
        Y: (n, ) numpy array containing the labels (number from 0-9) for each data point
        temp_parameter: the temperature parameter of softmax function
        alpha: the learning rate
        lambda_factor: the regularization constant
        k: number of label
        num_iterations: number of iterations to run gradient descent
    Returns:
        theta: (k, d) numpy array that is the final value of parameter theta
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    for i in range(num_iterations):
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)

    return theta


def predict(X, theta, temp_parameter):
    """
    Classifies the given dataset

    Args:
        X: 2D numpy array represents data points need to be classified
        theta: numpy array where row j represents the parameters of the model for label j
        temp_parameter: temperature parameter of softmax function
    Returns:
        Y: 1D numpy array containing the predicted result
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)

    return np.argmax(probabilities, axis=0)


def softmax_error(X, Y, theta, temp_parameter):
    """
    Calculates error on test dataset

    Args:
        X: 2D numpy array represents data points need to be classified
        Y: 1D numpy array represents the true label of data points
        theta: numpy array where row j represents the parameters of the model for label j
        temp_parameter: temperature parameter of softmax function
    Returns:
        test error
    """
    pred = predict(X, theta, temp_parameter)

    return 1- np.mean(pred == Y)
























