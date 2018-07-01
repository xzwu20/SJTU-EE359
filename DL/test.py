import torch
import model
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score
import time

def test(args, shared_model, inputs, labels, log):
    start_time = time.time()
    local_model = model.network(args, class_num = args.class_num, input_dim = inputs.shape[1])
    local_model.load_state_dict(shared_model.state_dict())
    if args.gpu:
        local_model = local_model.cuda()

    outputs = local_model(inputs)
    predict_labels = torch.argmax(outputs, dim = 1)
    if args.gpu:
        predict_labels = predict_labels.cpu()
    predict_labels = predict_labels.numpy()
    cnt = 0

    if args.gpu:
        labels = labels.cpu()
    labels = labels.numpy()
    for i in range(labels.shape[0]):
        if predict_labels[i] == labels[i]:
            cnt += 1
    accuracy = cnt / labels.shape[0]
    f1 = f1_score(list(labels), list(predict_labels), average='macro')

    log.info('Test time ' + \
                   time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + \
                   ',f1: %0.4f\t accuracy: %0.4f' % (f1, 100* accuracy))


    return accuracy, f1
