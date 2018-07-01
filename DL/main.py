import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from .model import network
from .test import test
import time
from .utils import setup_logger

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu', default=True, help = 'using GPU')
parser.add_argument('--load_model_path', type = str, default = "")
parser.add_argument('--lr', type = float, default = 0.0005, help = "learning rate")
parser.add_argument('--save_model_path', type = str, help = "path to save trained models", default = '../trained_models/')
parser.add_argument('--log_path', type = str, help="path to save logs",default = '../logs/')
parser.add_argument('--max_epoch', type = int, help = "max train epoch", default = 2000)
parser.add_argument('--l2', type = float, default = 0, help = "L2 norm factor")

data_path = "../processedDataset/"

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.DoubleTensor')
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # load data
    train_inputs = np.loadtxt(data_path+"train/train_microarray.txt", delimiter='\t')
    train_labels = np.loadtxt(data_path+"train/label_ds.txt", delimiter='\t').T.squeeze() # reduce label_index dimension from 2 to 1
    train_inputs = torch.from_numpy(train_inputs)
    train_labels = torch.from_numpy(train_labels)

    test_inputs = np.loadtxt(data_path+"test/test_microarray.txt", delimiter='\t')
    test_labels = np.loadtxt(data_path+"test/label_ds.txt", delimiter='\t').T.squeeze() # reduce label_index dimension from 2 to 1
    test_inputs = torch.from_numpy(test_inputs)
    test_labels = torch.from_numpy(test_labels)

    if args.gpu:
        train_inputs = train_inputs.cuda()
        train_labels = train_labels.cuda()
        test_inputs = test_inputs.cuda()
        test_labels = test_labels.cuda()


    # network part
    model = network(args, input_dim=train_inputs.shape[1], class_num= train_labels.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.l2)
    if args.load_model_path != "":
        opti_path = args.load_model_path + "_opti"
        if args.use.gpu:
            model.load_state_dict(torch.load(args.load_model_path))
            model.cuda()
        else:
            model.load_state_dict(torch.load(args.load_model_path))

    epoch = 0
    log_test = setup_logger(0, 'test_log', os.path.join(args.log_path, 'test_log.txt'))
    log_train = setup_logger(0, 'train_log', os.path.join(args.log_path, 'train_log.txt'))
    best_accuracy, best_f1, best_train_accuracy = 0.0, 0.0, 0.0
    early_stop_counter = 0
    loss_function = nn.CrossEntropyLoss()

    ftrain_accuracy = open((os.path.join(args.log_path, 'train_accuracy.txt')), "w")
    floss = open((os.path.join(args.log_path, 'loss.txt')), "w")
    # train
    while epoch<args.max_epoch:
        print('=====> Train at epoch %d, Learning rate %0.6f <=====' % (args.epoch, args.lr))
        start_time = time.time()
        log_train.info('Train time ' + time.strftime("%Hh %Mm %Ss",
                                               time.gmtime(time.time() - start_time)) + ', ' + 'Training started.')

        outputs = model(input)
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
            if predict_labels == train_labels_numpy[i]:
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
        print('train_accuracy:%0.3f, loss:%0.3f' %(train_accuracy, loss))
        log_train.info('Train time ' + \
                 time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + \
                 ', ' + 'loss: %0.4f'+'\t accuracy: %0.4f' % (loss))
        test_accuracy, test_f1 = test(args,model,test_inputs,test_labels, log_test)
        if test_accuracy>best_accuracy:
            best_accuracy = test_accuracy
            log_test.info("new best accuracy:%0.3f", best_accuracy*100)
        if test_f1 > best_f1:
            best_f1 = test_f1
            log_test.info("new best f1:%0.3f", best_f1)

        if epoch % 10 == 0:
            ftrain_accuracy.write('%d\t%0.3f\n' %(epoch, train_accuracy))
            ftrain_accuracy.write('%d\t%0.3f\n' % (epoch, loss))
        if early_stop_counter > 100:
            log_train.info('no improvement after 100 epochs, stop training')
            print('no improvement after 100 epochs, stop training')
            break

        epoch += 1
        optimizer.zero_grad()
    ftrain_accuracy.close()
    floss.close()
