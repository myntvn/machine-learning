import numpy as np

def project_onto_pc(X, pcs, n_components):
    """
    Given principal component vectors pcs, return a new data array in which each
    sample in X has been projected onto the first n_components principal component
    """
    # center the data
    X = center_data(X)

    # select n_components from pcs
    selected_pcs = pcs[:, :n_components]

    return np.matmul(X, selected_pcs)


def center_data(X):
    """
    Returns a centered version of the data, where each feature has mean 0,
    by subtracting the mean on each feature

    Args:
        X: 2D numpy array represents data points
    Returns:
        the same shape 2D numpy array which has been subtracted the mean from each of features
    """
    means = X.mean(axis=0)

    return X - means


def principal_components(X):
    """
    Return the principal component vectors of the data

    Args:
        X: n x d numpy array represents n data points each with d features
    Returns:
        d x d numpy array whose columns are the principal components sorted in
        descending order of eigenvalue magnitude
    """
    # center the data - mean 0 on each feature
    X = center_data(X)

    # calculate scatter matrix
    scatter_matrix = np.dot(X.T, X)

    # compute eigen values and eigen vectors
    eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix)

    # sort eigenvector by eigenvalue
    idx = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_vectors
