from utils import *
from age_filter import age_group_scaling

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood_m1(data, theta, beta, age_scale):
    """ Compute the negative log-likelihood, according to model 1 extension.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :return: float
    """
    log_lklihood = 0.
    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]

        theta_i = theta[i]
        beta_j = beta[j]
        diff = age_scale[i] * theta_i - beta_j
        log_lklihood += cij * diff - np.log(1 + np.exp(diff))
    return -log_lklihood


def neg_log_likelihood_m2(data, theta, beta, p, k):
    """ Compute the negative log-likelihood, according to model 2 extension.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param p: Scalar, probability of getting question correct via guess
    :param k: Vector, how discriminative the question is
    :return: float
    """
    log_lklihood = 0.
    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]
        k_j = k[j]

        theta_i = theta[i]
        beta_j = beta[j]
        diff = k_j * (theta_i - beta_j)
        log_lklihood += cij * np.log(p + (1 - p) * sigmoid(diff)) + \
            (1 - cij) * np.log(1 - p - (1 - p) * sigmoid(diff))
    return -log_lklihood


def neg_log_likelihood_m3(data, theta, beta, age_scale, p, k):
    """ Compute the negative log-likelihood, according to model 3 extension.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :param p: Scalar, probability of getting question correct via guess
    :param k: Vector, how discriminative the question is
    :return: float
    """
    log_lklihood = 0.
    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]
        k_j = k[j]

        theta_i = theta[i]
        beta_j = beta[j]
        diff = k_j * (age_scale[i] * theta_i - beta_j)
        log_lklihood += cij * np.log(p + (1 - p) * sigmoid(diff)) + \
            (1 - cij) * np.log(1 - p - (1 - p) * sigmoid(diff))
    return -log_lklihood


def update_param_m1(data, lr, theta, beta, age_scale):
    """ Update theta and beta using gradient descent. Used for model 1.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :return: tuple of vectors
    """
    diff_theta_beta = np.expand_dims(np.multiply(age_scale, theta), axis=1) - \
        np.expand_dims(beta, axis=0)
    sig = sigmoid(diff_theta_beta)

    grad_theta = np.zeros_like(diff_theta_beta)
    grad_beta = np.zeros_like(diff_theta_beta)

    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]

        grad_theta[i, j] = age_scale[i] * (cij - sig[i, j])
        grad_beta[i, j] = sig[i, j] - cij

    theta = theta + lr * np.sum(grad_theta, axis=1)
    beta = beta + lr * np.sum(grad_beta, axis=0)
    return theta, beta


def update_param_m2(data, lr, theta, beta, p, k):
    """ Update theta and beta using gradient descent. Used for model 2.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param p: Scalar, probability of getting question correct via guess
    :param k: Vector, how discriminative the question is

    :return: tuple of vectors
    """
    diff_theta_beta = (np.expand_dims(theta, axis=1) - np.expand_dims(beta, axis=0)) * k
    sig = sigmoid(diff_theta_beta)

    grad_theta = np.zeros_like(diff_theta_beta)
    grad_beta = np.zeros_like(diff_theta_beta)
    grad_k = np.zeros_like(diff_theta_beta)

    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]
        k_j = k[j]

        grad_theta[i, j] = cij * k_j * ((1-p) * sig[i, j]) / (np.exp(diff_theta_beta[i, j]) + p) - \
            k_j * (1-cij) * sig[i, j]

        grad_beta[i, j] = cij * k_j * (p-1) * sig[i, j] / (np.exp(diff_theta_beta[i, j]) + p) + \
            (1-cij) * k_j * sig[i, j]

        grad_k[i, j] = cij * (1-p) * (theta[i] - beta[j]) * sig[i, j] / (np.exp(diff_theta_beta[i, j]) + p) - \
            (1-cij) * (theta[i] - beta[j]) * sig[i, j]

    theta = theta + lr * np.sum(grad_theta, axis=1)
    beta = beta + lr * np.sum(grad_beta, axis=0)
    k = k + lr * np.sum(grad_k, axis=0)
    return theta, beta, k


def update_param_m3(data, lr, theta, beta, age_scale, p, k):
    """ Update theta and beta using gradient descent. Used for model 3.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :param p: Scalar, probability of getting question correct via guess
    :param k: Vector, how discriminative the question is

    :return: tuple of vectors
    """
    diff_theta_beta = (np.expand_dims(theta * age_scale, axis=1) - np.expand_dims(beta, axis=0)) * k
    sig = sigmoid(diff_theta_beta)

    grad_theta = np.zeros_like(diff_theta_beta)
    grad_beta = np.zeros_like(diff_theta_beta)
    grad_k = np.zeros_like(diff_theta_beta)

    for ind in np.arange(len(data["is_correct"])):
        i = data["user_id"][ind]
        j = data["question_id"][ind]
        cij = data["is_correct"][ind]
        k_j = k[j]

        grad_theta[i, j] = cij * k_j * (1-p) * age_scale[i] * sig[i, j] / (np.exp(diff_theta_beta[i, j]) + p) - \
            k_j * (1-cij) * sig[i, j] * age_scale[i]

        grad_beta[i, j] = cij * k_j * (p-1) * sig[i, j] / (np.exp(diff_theta_beta[i, j]) + p) + \
            (1-cij) * k_j * sig[i, j]

        grad_k[i, j] = cij * (1-p) * (age_scale[i] * theta[i] - beta[j]) * sig[i, j] / (np.exp(diff_theta_beta[i, j]) + p) - \
            (1-cij) * (age_scale[i] * theta[i] - beta[j]) * sig[i, j]

    theta = theta + lr * np.sum(grad_theta, axis=1)
    beta = beta + lr * np.sum(grad_beta, axis=0)
    k = k + lr * np.sum(grad_k, axis=0)
    return theta, beta, k


def irt_m1(data, val_data, age_scale, lr, iterations):
    """ Train IRT model 1.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :param lr: float, learning rate
    :param iterations: int, numbers of iterations
    :return: (theta, beta, val_acc_lst, neg_lld_lst, val_lld_lst)
    """
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    neg_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood_m1(data, theta=theta, beta=beta, age_scale=age_scale)
        score = evaluate_m1(data=val_data, theta=theta, beta=beta, age_scale=age_scale)
        val_acc_lst.append(score)
        neg_lld_lst.append(neg_lld)
        val_lld_lst.append(neg_log_likelihood_m1(val_data, theta=theta, beta=beta, age_scale=age_scale))
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_param_m1(data, lr, theta, beta, age_scale)

    return theta, beta, val_acc_lst, neg_lld_lst, val_lld_lst


def irt_m2(data, val_data, p, lr, iterations):
    """ Train IRT model 2.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param p: Scalar, probability of getting question correct via guess
    :param lr: float, learning rate
    :param iterations: int, numbers of iterations
    :return: (theta, beta, val_acc_lst, neg_lld_lst, val_lld_lst)
    """
    theta = np.zeros(542)
    beta = np.zeros(1774)
    k = np.ones(1774)

    val_acc_lst = []
    neg_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood_m2(data, theta=theta, beta=beta, p=p, k=k)
        score = evaluate_m2(data=val_data, theta=theta, beta=beta, p=p, k=k)
        val_acc_lst.append(score)
        neg_lld_lst.append(neg_lld)
        val_lld_lst.append(neg_log_likelihood_m2(val_data, theta=theta, beta=beta, p=p, k=k))
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, k = update_param_m2(data, lr, theta, beta, p, k)

    return theta, beta, k, val_acc_lst, neg_lld_lst, val_lld_lst


def irt_m3(data, val_data, age_scale, p, lr, iterations):
    """ Train IRT model 3.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :param p: Scalar, probability of getting question correct via guess
    :param lr: float, learning rate
    :param iterations: int, numbers of iterations
    :return: (theta, beta, val_acc_lst, neg_lld_lst, val_lld_lst)
    """
    theta = np.zeros(542)
    beta = np.zeros(1774)
    k = np.ones(1774)

    val_acc_lst = []
    neg_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood_m3(data, theta=theta, beta=beta, age_scale=age_scale, p=p, k=k)
        score = evaluate_m3(data=val_data, theta=theta, beta=beta, age_scale=age_scale, p=p, k=k)
        val_acc_lst.append(score)
        neg_lld_lst.append(neg_lld)
        val_lld_lst.append(neg_log_likelihood_m3(val_data, theta=theta, beta=beta, age_scale=age_scale, p=p, k=k))
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, k = update_param_m3(data, lr, theta, beta, age_scale, p, k)

    return theta, beta, k, val_acc_lst, neg_lld_lst, val_lld_lst


def evaluate_m1(data, theta, beta, age_scale):
    """ Evaluate model 1 given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (age_scale[u] * theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def evaluate_m2(data, theta, beta, p, k):
    """ Evaluate model 1 given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param p: Scalar, probability of getting question correct via guess
    :param k: Vector, how discriminative the question is
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = ((theta[u] - beta[q]) * k[q]).sum()
        p_a = sigmoid(x)
        p_a = p + (1-p) * p_a
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def evaluate_m3(data, theta, beta, age_scale, p, k):
    """ Evaluate model 1 given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param age_scale: A list that contains scales for all users (in sorted
    user_id)
    :param p: Scalar, probability of getting question correct via guess
    :param k: Vector, how discriminative the question is
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = ((age_scale[u] * theta[u] - beta[q]) * k[q]).sum()
        p_a = sigmoid(x)
        p_a = p + (1-p) * p_a
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    age_scale = age_group_scaling()

    lr = 1e-2
    num_iteration = 50
    print("Model 1")
    theta_m1, beta_m1, val_acc_lst_m1, neg_lld_lst_m1, val_lld_lst_m1 = irt_m1(train_data, val_data, age_scale, lr, num_iteration)
    print("End Model 1\n")
    print("Model 2")
    theta_m2, beta_m2, k_m2, val_acc_lst_m2, neg_lld_lst_m2, val_lld_lst_m2 = irt_m2(train_data, val_data, 0.25, lr, num_iteration)
    print("End Model 2\n")
    print("Model 3")
    theta_m3, beta_m3, k_m3, val_acc_lst_m3, neg_lld_lst_m3, val_lld_lst_m3 = irt_m3(train_data, val_data, age_scale, 0.25, lr, num_iteration)
    print("End Model 3\n")

    print("Validation accuracy (model 1): {}".format(val_acc_lst_m1[-1]))
    test_acc = evaluate_m1(test_data, theta_m1, beta_m1, age_scale)
    print("Test accuracy (model 1): {}".format(test_acc))

    print("Validation accuracy (model 2): {}".format(val_acc_lst_m2[-1]))
    test_acc = evaluate_m2(test_data, theta_m1, beta_m1, 0.25, k_m2)
    print("Test accuracy (model 2): {}".format(test_acc))

    print("Validation accuracy (model 3): {}".format(val_acc_lst_m3[-1]))
    test_acc = evaluate_m3(test_data, theta_m1, beta_m1, age_scale, 0.25, k_m3)
    print("Test accuracy (model 3): {}".format(test_acc))


if __name__ == "__main__":
    main()
