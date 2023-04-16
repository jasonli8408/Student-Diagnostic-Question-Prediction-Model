from utils import *

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]

        theta_i = theta[i]
        beta_j = beta[j]
        diff = theta_i - beta_j
        log_lklihood += cij * diff - np.log(1 + np.exp(diff))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    diff_theta_beta = np.expand_dims(theta, axis=1) - np.expand_dims(beta, axis=0)
    sig = sigmoid(diff_theta_beta)

    grad_theta = np.zeros_like(diff_theta_beta)
    grad_beta = np.zeros_like(diff_theta_beta)

    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]

        grad_theta[i, j] = cij - sig[i, j]
        grad_beta[i, j] = sig[i, j] - cij

    theta = theta + lr * np.sum(grad_theta, axis=1)
    beta = beta + lr * np.sum(grad_beta, axis=0)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    neg_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_lst.append(neg_lld)
        val_lld_lst.append(neg_log_likelihood(val_data, theta=theta, beta=beta))
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 1e-2
    num_iteration = 50
    theta, beta, val_acc_lst, neg_lld_lst, val_lld_lst = irt(train_data, val_data, lr, num_iteration)

    # q2b plotting only
    """fig, ax = plt.subplots()
    ax.set_title("Negative Log-likelihood vs # iterations")
    num_iter = np.arange(num_iteration)
    ax.plot(num_iter, neg_lld_lst, label="train")
    ax.plot(num_iter, val_lld_lst, label="validation")
    ax.set_xlabel("iterations")
    ax.set_ylabel("Negative Log-likelihood")
    ax.legend()
    plt.show()"""

    print("Validation accuracy: {}".format(val_acc_lst[-1]))
    test_acc = evaluate(test_data, theta, beta)
    print("Test accuracy: {}".format(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # select 3 questions at random
    np.random.seed(311)
    q_ids = np.sort(np.random.randint(0, 1774, 3))

    # evenly spaced theta value between -5 to 5, used for plotting
    range_theta = np.linspace(-5., 5., 100)

    # for plotting
    fig, ax = plt.subplots()
    ax.set_title("p(c_ij = 1) over theta values")
    ax.set_xlabel("theta")
    ax.set_ylabel("p(c_ij = 1)")

    for q_id in q_ids:
        diff = range_theta - beta[q_id]
        prob = sigmoid(diff)

        ax.plot(range_theta, prob, label="Q{0} with beta {1}".format(q_id, round(beta[q_id], 2)))

    ax.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
