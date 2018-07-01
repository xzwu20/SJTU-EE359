import torch
from .model import network
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F
import time

def test(args, shared_model, inputs, labels, log):
    start_time = time.time()
    log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Start testing.')
    local_model = network(args)
    if args.gpu:
        local_model = local_model.cuda()

    outputs = local_model(inputs)
    if args.gpu:
        outputs = outputs.cpu()
    outputs = outputs.numpy()
    predict_labels = np.argmax(outputs, axis = 1)
    cnt = 0

    if args.gpu:
        labels = labels.cpu()
    labels = labels.numpy()
    for i in range(labels.shape[0]):
        if predict_labels == labels[i]:
            cnt += 1
    accuracy = cnt / labels.shape[0]
    log.info('Overall f1 score = %0.4f' % (f1_score(list(labels), list(predict_labels), average='weighted')))
    log.info('Overall accuracy = %0.2f%%' % (100 * accuracy))

    return accuracy
