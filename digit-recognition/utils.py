import pickle, gzip
import numpy as np

def load_mnist_data():
    """
    Read mnist dataset from file

    Returns:
        train_x: 2D numpy array where each row contains features (raw pixel values) of an image
        train_y: 1D numpy array where each row is label of an image (number from 0-9)
        test_x: 2D numpy array where each row contains features (raw pixel values) of an image
        test_y: 1D numpy array where each row is label of an image (number from 0-9)
    """

    file_name = 'dataset/mnist.pkl.gz'

    # pickle.load return ((train_x, train_y), (valid_x, valid_y), (test_x, test_y))
    with gzip.open(file_name, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    train_x, train_y = train_set
    valid_x, valid_y = valid_set

    # concatenation valid set to train set
    train_x = np.vstack((train_x, valid_x))
    train_y = np.append(train_y, valid_y)

    test_x, test_y = test_set

    return (train_x, train_y, test_x, test_y)
