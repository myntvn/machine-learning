import numpy as np
from sklearn.svm import LinearSVC

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Train a linear SVM for binary classification

    Args:
        train_x: 2D numpy array containing data points for training
        train_y: 1D numpy array containing label for each data point
        test_x: 2D numpy array containing data points for testing
    Returns:
        pred_test_y: 1D numpy array containing predicted label (0 or 1) for each test data point
    """

    clf = LinearSVC(C=0.1, random_state=0)

    clf.fit(train_x, train_y)

    pred_test_y = clf.predict(test_x)

    return pred_test_y


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains linear SVM for multiclass classification using a one-vs-rest strategy
     Args:
        train_x: 2D numpy array containing data points for training
        train_y: 1D numpy array containing label for each data point
        test_x: 2D numpy array containing data points for testing
    Returns:
        pred_test_y: 1D numpy array containing predicted label (int) for each test data point
    """
    clf = LinearSVC(C=0.1, random_state=0)

    clf.fit(train_x, train_y)

    pred_test_y = clf.predict(test_x)

    return pred_test_y


def svm_error(test_y, pred):
    """
    Computes error by zero-one loss function

    Args:
        test_y: 1D numpy array containing true label
        pred: 1D numpy array containing the predicted label

    Returns:
        zero-one error
    """

    return 1 - np.mean(pred == test_y)
