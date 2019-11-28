import numpy as np
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import gzip # pour décompresser les données
import pickle # pour désérialiser les données


if __name__ == '__main__':
    #Load data
    data = pickle.load(gzip.open('mnist.pkl.gz'),encoding='latin1')
    train_data = torch.Tensor(data[0][0])
    train_data_label = torch.Tensor(data[0][1])
    test_data = torch.Tensor(data[1][0])
    test_data_label = torch.Tensor(data[1][1])
    
    learning_rate = .001
    epochs = 1
    nb_neurons = 10
    nb_entries = 785
    w = np.random.randn(nb_entries, nb_neurons)

    for i in range(epochs):
        for j in range(len(train_data)):
            x = train_data[j]
            t = train_data_label[j]
            x = torch.cat((x, torch.Tensor([1])), 0)
            y = torch.Tensor(np.dot(x, w))
            dw = learning_rate * x * (t - y)
            w += dw
