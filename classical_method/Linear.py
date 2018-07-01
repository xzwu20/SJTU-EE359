import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

data_path = "../processedDataset/"

if __name__ == '__main__':

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

    for factor in [0.01, 0.1, 1.0, 10.0, 100.0]:
        # LR = LogisticRegression(C = factor)
        # LR = LogisticRegression(C=factor, penalty= 'l1', tol = 0.001)
        LR = LinearSVC(C = factor, loss = 'hinge', tol = 1e-4)
        LR.fit(train_inputs, train_labels)
        predict_labels = LR.predict(test_inputs)
        accuracy = LR.score(test_inputs, test_labels, sample_weight = None)
        print("factor:", factor)
        print(accuracy)
        print(f1_score(test_labels, predict_labels, average='binary'))