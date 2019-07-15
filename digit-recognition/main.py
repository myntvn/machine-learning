from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *


# load mnist data
print('Load minist data')
train_x, train_y, test_x, test_y = load_mnist_data()


def run_linear_regression(lambda_factor=1):
    """
    Trains linear regression, classifies and computes test error on test set

    Args:
        lambda_factor: the regularization constant
    Returns:
        test error
    """
    # run linear regression on the training dataset
    theta = closed_form(train_x, train_y, lambda_factor)

    # calculate error on the test set
    error = linear_reg_error(test_x, test_y, theta)

    print('Linear Regression test error: ', error)


def run_svm_one_vs_rest():
    """
    Trains svm, classifies and computes test error on test set

    Return:
        test error
    """
    # set label which is different from 0 to 1
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1

    # train and predict
    pred = one_vs_rest_svm(train_x, train_y, test_x)

    # calculate the error
    error = svm_error(test_y, pred)

    return error


def run_multi_class_svm():
    """
    Trains svm, classifies and computes test error on test set

    Return:
        test error
    """
    # train and predict
    pred = multi_class_svm(train_x, train_y, test_x)

    # calculate the error
    error = svm_error(test_y, pred)

    return error


def run_softmax(temp_parameter=1):
    """
    Trains softmax, classifies and computes error on test data

    Args:
        temp_parameter: the temperature parameter of softmax function
    Returns:
        test error
    """
    # training the model on training set
    theta = softmax_regression(train_x, train_y, temp_parameter,
            alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)

    # predict testing set and compute error
    test_error = softmax_error(test_x, test_y, theta, temp_parameter)

    return test_error


def run_softmax_mod3(temp_parameter=1):
    """
    Trains softmax regression on digit (mod 3)

    Args:
        temp_parameter: the temperature parameter of softmax function
    Returns:
        test error
    """
    # get mod3 label for training and testing set
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)

    # train the model on training set
    theta = softmax_regression(train_x, train_y_mod3, temp_parameter,
            alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)

    # predict testing set and compute error
    test_error = softmax_error_mod3(test_x, test_y_mod3, theta, temp_parameter)

    return test_error


def softmax_with_pca(temp_parameter=1):
    n_components = 18

    #get principal components
    pcs = principal_components(train_x)

    # apply pca on train set and test set
    train_pca = project_onto_pc(train_x, pcs, n_components)
    test_pca = project_onto_pc(test_x, pcs, n_components)

    # train the softmax regression by the data which was applies pca
    theta = softmax_regression(train_pca, train_y, temp_parameter,
            alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)

    # predict testing set (applied pca) and compute error
    test_error = softmax_error(test_pca, test_y, theta, temp_parameter)

    return test_error


if __name__ == '__main__':
    # print('Load minist data')
    # train_x, train_y, test_x, test_y = load_mnist_data()

    # print('run linearr regression')
    # run_linear_regression(lambda_factor=1)

    # print('run svm')
    # print('SVM one vs rest error: ', run_svm_one_vs_rest())

    # print('run multi-class svm')
    # print('Multi-class SVM test error: ', run_multi_class_svm())

    # print('run softmax')
    # print('softmax test error: ', run_softmax(temp_parameter = 1))

    # print('run softmax mod3')
    # print('softmax test error mod3: ', run_softmax_mod3(temp_parameter = 1))

    print('run softmax with pca applied data')
    print('softmax test error with pca applied data: ', softmax_with_pca())











