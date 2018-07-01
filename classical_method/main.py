import os
import sys
import argparse
import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--method', type = str, default="SVM", help = 'using GPU')
parser.add_argument('--lr', type = float, default = 0.005, help = "learning rate")
parser.add_argument('--l2', type = float, default = 0, help = "L2 norm factor")


data_path = "../processedDataset/"

if __name__ == '__main__':
    args = parser.parse_args()

    # load data
    inputs=np.loadtxt(data_path+"microarray_0.9.txt",delimiter='\t')
    m = inputs.shape[0]
    label = np.zeros(m, dtype=np.int)
    with open(data_path+"label_mt.txt", "r") as fin1:
        index = 0
        for line in fin1:
            label[index] = int(line[:-1])
            index += 1

    # split train and test dataset
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, label, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(train_inputs, train_labels)
    predict_labels = knn.predict(test_inputs)
    accuracy = knn.score(test_inputs, test_labels, sample_weight = None)
    print(accuracy)
    # print(precision_score(test_labels, predict_labels, average = 'micro'))
    print(f1_score(test_labels, predict_labels, average = 'micro'))
