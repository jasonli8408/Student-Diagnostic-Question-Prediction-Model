from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs

        # Q3B
        g_out = torch.sigmoid(self.g(inputs))
        h_out = torch.sigmoid(self.h(g_out))
        out = h_out
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. --> In line 122
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    loss_lst = []
    acc_lst = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            # Added squared error regularizer
            loss = torch.sum((output - target) ** 2.) + (model.get_weight_norm() * lamb / 2)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        loss_lst.append(train_loss)
        acc_lst.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        
    return valid_acc, loss_lst, acc_lst
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    # Q3A
    # 1. ALS fix all other variables and perform gradient descent on 1 unfixed variable, in an alternative fasion, 
    #       where as neural network perform backward propagation without explictly "fixing" variables since it just
    #       does back prop in the backward order
    # 2. ALS have to be linear, where as neural networks can be non-linear given non-linear activation functions
    # 3. ALS may be more efficient than neural network due to linear nature
    
    # Q3C
    # Set model hyperparameters.
    print("Starting Q3C")
    k_lst = [10, 50, 100, 200, 500]
    
    # Set optimization hyperparameters, tuned via multiple trials
    lr = 0.005
    num_epoch = 100
    lamb = 0
    
    max_so_far = float("-inf")
    k_star = None
    best_loss = None
    best_acc = None
    
    for k in k_lst:
        model = AutoEncoder(train_matrix.shape[1], k)
        print(f"Starting to test k = {k}")
        final_valid_acc, loss_lst, acc_lst = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        
        if final_valid_acc > max_so_far:
            k_star = k
            best_loss = loss_lst
            best_acc = acc_lst
            max_so_far = final_valid_acc
            
    print(f"We choose k* to be {k_star} with validation accuracy of {max_so_far}")
        
    # Q3D
    print("Starting Q3D")
    epochs = [x for x in range(1, num_epoch + 1)]

    model = AutoEncoder(train_matrix.shape[1], k_star)

    # Compute epoch vs train_loss graph
    plt.title(f"Epoch vs Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.plot(epochs, best_loss)
    plt.show()

    # Compute epoch vs valid_acc graph
    plt.title(f"Epoch vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.plot(epochs, best_acc)
    plt.show()

    max_acc = max(best_acc)
    best_epoch = best_acc.index(max_acc) + 1

    print(f"We will use best epoch = {best_epoch} with validation accuracy of {max_acc} for test set")

    # Find test accuracy used the epoch that gave best validation accuracy
    test_acc, loss_lst, acc_lst = train(model, lr, lamb, train_matrix, zero_train_matrix, test_data, best_epoch)
    print(f"Test data achieved accuracy of {test_acc}")
    
    # Q3E
    print("Starting Q3E")
    lamb_lst = [0.001, 0.01, 0.1, 1]

    max_acc_so_far = float("-inf")
    best_lamb = None

    for lamb_val in lamb_lst:
        final_valid_acc, loss_lst, acc_lst = train(model, lr, lamb_val, train_matrix, zero_train_matrix, valid_data, best_epoch)
        
        if final_valid_acc > max_acc_so_far:
            best_lamb = lamb_val
            max_acc_so_far = final_valid_acc      

    test_acc_final, loss_lst, acc_lst = train(model, lr, best_lamb, train_matrix, zero_train_matrix, test_data, best_epoch)

    print(f"The best lambda value is {best_lamb}, achieving validation accuracy of {max_acc_so_far} and test accuracy of {test_acc_final}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Results: 
    # Q3C
    #   We choose k* to be 50 with validation accuracy of 0.6837425910245555
    # Q3D:
    #   We will use best epoch = 75 with validation accuracy of 0.6885407846457804 for test set
    #   Test data achieved accuracy of 0.6819079875811459
    # Q3E:
    #   The best lambda value is 0.001, achieving validation accuracy of 0.6850127011007621 and test accuracy of 0.6751340671747107
    #   It seems like regularizer penalty made the model perform worse.


if __name__ == "__main__":
    main()
