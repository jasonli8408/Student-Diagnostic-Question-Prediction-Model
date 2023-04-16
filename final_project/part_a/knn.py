import numpy as np
from sklearn.impute import KNNImputer
from utils import *

# for plotting
from matplotlib import pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    accuracies = []
    k_lst = np.arange(1, 30, 5)
    for k in k_lst:
        # by user
        #accuracies.append(knn_impute_by_user(sparse_matrix, val_data, k))

        # by item
        accuracies.append(knn_impute_by_item(sparse_matrix, val_data, k))

    # for plotting (part a, c)
    #fig, ax = plt.subplots()
    # by user
    #ax.set_title("Accuracy vs k (by user)")
    # by item
    #ax.set_title("Accuracy vs k (by item)")
    #ax.plot(k_lst, accuracies)
    #ax.set_xlabel("k")
    #ax.set_ylabel("Accuracy")
    #ax.set_xticks(k_lst)
    #plt.show()

    # evaluation on test data with k* (part b, c)
    # by user
    #k_star = 11
    #print("Evaluating knn on k = {} on the test data:".format(k_star))
    #knn_impute_by_user(sparse_matrix, test_data, k_star)
    # by item
    k_star = 21
    print("Evaluating knn on k = {} on the test data:".format(k_star))
    knn_impute_by_item(sparse_matrix, test_data, k_star)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
