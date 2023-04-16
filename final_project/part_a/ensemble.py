from neural_network import *
from utils import *
from torch.autograd import Variable

import numpy as np
import torch
import torch.utils.data

    
def evaluate_nn(model, lr, lamb, train_data, zero_train_data, test_data, num_epoch):
    """
    refer to neural_network.py, train the data using given optimized parameters, and return output

    Args:
        :param model: Module
        :param lr: float
        :param lamb: float
        :param train_data: 2D FloatTensor
        :param zero_train_data: 2D FloatTensor
        :param test_data: Dict
        :param num_epoch: int
    """
    
    train(model, lr, lamb, train_data, zero_train_data, test_data, num_epoch)
    res = []
    for i, u in enumerate(test_data['user_id']):
        input1 = Variable(zero_train_data[u]).unsqueeze(0)
        output1 = model(input1)
        res.append(output1[0][test_data['question_id'][i]].item())
    return res


def calculate_accuracy(source, data_input):
    """
    calculate the accuracy of the source compared to data_input and return it

    Args:
        source (dict array): source data
        data_input (dict array): compared data (test_data / valid_data)
    """
    
    acc = 0
    leng = len(source)
    for i, u in enumerate(source):
        if u >= 0.5 and data_input['is_correct'][i]:
            acc += 1
        if u < 0.5 and not data_input['is_correct'][i]:
            acc += 1
    return acc / leng
    
    
def main():
    train_data = load_train_sparse('../data').toarray()
    valid_data = load_valid_csv('../data')
    test_data = load_public_test_csv('../data')
    prediction = []
    lr = 0.005
    num_epoch = 75
    lamb = 0.001
    k = 50
    for i in range(3): 
        #model = AutoEncoder(train_data.shape[1], k)
        row = train_data.shape[0]
        rand = np.random.randint(0, row, row)
        trainm = np.zeros(train_data.shape)
        print(test_data)
        for j in range(row):
            index = rand[j]
            print(j, index)
            trainm[j] = train_data[index]
        trainz = trainm.copy()
        trainz[np.isnan(trainm)] = 0
        trainz = torch.FloatTensor(trainz)
        trainm = torch.FloatTensor(trainm)
        print(f"Model {i}")
        model = AutoEncoder(trainz.shape[1], k)
        prediction.append(evaluate_nn(model, lr, lamb, trainm, trainz, test_data, num_epoch))
    sum = np.sum(prediction, axis=0)
    source = [k / 3 for k in sum]
    valid_accuracy = calculate_accuracy(source, valid_data)
    print(f"valid accuracy: {valid_accuracy}")
    test_accuracy = calculate_accuracy(source, test_data)
    print(f"test accuracy: {test_accuracy}")
        
        
if __name__ == "__main__":
    main()
