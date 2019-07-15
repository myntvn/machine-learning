import numpy as np

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X: 2D numpy array represents data points
        Y: 1D numpy array represents the label (number from 0-9) for each data point
        lambda_factor: the regularization constant

    Returns:
        theta: 1D numpy array containing the weights of linear regression
    """

    x_transpose = np.transpose(X)

    feature_size = X.shape[1]

    # identity matrix
    i_matrix = np.identity(feature_size)

    a = x_transpose.dot(X) + lambda_factor * i_matrix
    a_inverse = np.linalg.inv(a)

    b = x_transpose.dot(Y)

    return a_inverse.dot(b)


def linear_reg_error(X, Y, theta):
    """
    Compute error on the test set

    Args:
        X: 2D numpy array represents data points
        Y: 1D numpy array represents the label (number from 0-9) for each data point
        theta: 1D numpy array containing the weights which was obtain from training set
    Returns:
        error: error on the test set
    """

    predict = np.round(np.dot(X, theta))

    predict[predict < 0] = 0
    predict[predict > 9] = 9

    return 1 - np.mean(predict == Y)
