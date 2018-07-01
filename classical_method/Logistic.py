import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LinearRegression


data_path = "../processedDataset/"

if __name__ == '__main__':
    # load data
    inputs=np.loadtxt(data_path+"microarray_0.9.txt",delimiter='\t')
    m = inputs.shape[0]
    print(inputs.shape[1])
    label = np.zeros(m, dtype=np.int)
    with open(data_path+"label_ds.txt", "r") as fin1:
        index = 0
        for line in fin1:
            label[index] = int(line[:-1])
            index += 1

    # split train and test dataset
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, label, test_size=0.2, random_state=42)

    LR = LinearRegression()
    LR.fit(train_inputs, train_labels)
    accuracy = LR.score(test_inputs, test_labels, sample_weight = None)
    print(accuracy)
