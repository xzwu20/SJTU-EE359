import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import model
import test
import utils
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu', default=False, help = 'using GPU')
parser.add_argument('--load_model_path', type = str, default = "")
parser.add_argument('--lr', type = float, default = 0.005, help = "learning rate")  # default 0.005
parser.add_argument('--save_model_path', type = str, help = "path to save trained models", default = '../trained_models/')
parser.add_argument('--log_path', type = str, help="path to save logs",default = '../logs/')
parser.add_argument('--max_epoch', type = int, help = "max train epoch", default = 2000)
parser.add_argument('--l2', type = float, default = 1e-2, help = "L2 norm factor") # default 1e-4
parser.add_argument('--class_num', type = int, default = 79, help = "the number of class")


data_path = "../processedDataset/"

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.DoubleTensor')
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # load data
    inputs=np.loadtxt(data_path+"microarray_0.9.txt",delimiter='\t')
    m = inputs.shape[0]
    label = np.zeros(m, dtype=np.int)
    with open(data_path+"label_ds.txt", "r") as fin1:
        index = 0
        for line in fin1:
            label[index] = int(line[:-1])
            index += 1

    # split train and test dataset
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, label, test_size=0.2, random_state=42)

    train_inputs = torch.from_numpy(train_inputs)
    train_labels = torch.from_numpy(train_labels).long()
    test_inputs = torch.from_numpy(test_inputs)
    test_labels = torch.from_numpy(test_labels).long()

    if args.gpu:
        train_inputs = train_inputs.cuda()
        train_labels = train_labels.cuda()
        test_inputs = test_inputs.cuda()
        test_labels = test_labels.cuda()

    epoch = 0
    lr = args.lr
    # network part
    model = model.network(args, input_dim=train_inputs.shape[1], class_num= args.class_num)
    print(train_inputs.shape, train_labels.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.l2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr = args.lr, weight_decay = args.l2)
    if args.load_model_path != "":
        opti_path = args.load_model_path + "_opti"
        model.load_state_dict(torch.load(args.load_model_path))
    if args.gpu:
        model.cuda()

    epoch = 0
    log_test = utils.setup_logger(0, 'test_log', os.path.join(args.log_path, 'ds_test_log.txt'))
    log_train = utils.setup_logger(0, 'train_log', os.path.join(args.log_path, 'ds_train_log.txt'))
    best_accuracy, best_f1, best_train_accuracy = 0.0, 0.0, 0.0
    early_stop_counter = 0
    loss_function = nn.CrossEntropyLoss()

    ftrain_accuracy = open((os.path.join(args.log_path, 'l2_'+str(args.l2)+'ds_train_accuracy.txt')), "w")
    floss = open((os.path.join(args.log_path, 'l2_'+str(args.l2)+'ds_loss.txt')), "w")
    ftest = open((os.path.join(args.log_path, 'l2_'+str(args.l2)+'_ds_test_accuracy.txt')), "w")
    # train
    while epoch<args.max_epoch:
        print('=====> Train at epoch %d, Learning rate %0.6f <=====' % (epoch, lr))
        start_time = time.time()
        log_train.info('Train time ' + time.strftime("%Hh %Mm %Ss",
                                               time.gmtime(time.time() - start_time)) + ', ' + 'Training started.')

        outputs = model(train_inputs)
        loss = loss_function(outputs, train_labels)
        loss.backward()
        optimizer.step()

        predict_labels = torch.argmax(outputs, dim = 1)

        if args.gpu:
            predict_labels = predict_labels.cpu()
            train_labels = train_labels.cpu()
        predict_labels = predict_labels.numpy()
        train_labels_numpy = train_labels.numpy()
        cnt = 0
        for i in range(train_labels.shape[0]):
            if predict_labels[i] == train_labels_numpy[i]:
                cnt += 1
        train_accuracy = 100.0 * cnt / train_labels.shape[0]
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if args.gpu:
            train_labels = train_labels.cuda()
            loss = loss.cpu()
        log_train.info('Train time ' + \
                 time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + \
                 ',loss: %0.4f\t accuracy: %0.4f' % (loss, train_accuracy))
        test_accuracy, test_f1 = test.test(args, model, test_inputs, test_labels, log_test)
        if test_accuracy>best_accuracy:
            best_accuracy = test_accuracy
            log_test.info("new best accuracy:%0.3f", best_accuracy*100)
        if test_f1 > best_f1:
            best_f1 = test_f1
            log_test.info("new best f1:%0.3f", best_f1)

        if epoch % 5 == 0:
            ftrain_accuracy.write('%d\t%0.3f\n' %(epoch, train_accuracy))
            floss.write('%d\t%0.3f\n' % (epoch, loss))
            ftest.write('%d\t%0.3f\n' %(epoch, test_accuracy))
        if early_stop_counter > 100:
            log_train.info('no improvement after 100 epochs, stop training')
            print('no improvement after 100 epochs, stop training')
            break

        epoch += 1
        optimizer.zero_grad()
    ftrain_accuracy.close()
    floss.close()
