import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # weight_shape = list(m.weight.data.size())
        # fan_in = weight_shape
        nn.init.xavier_normal(m.weight)
        # nn.init.normal_(m.weight)
        nn.init.constant(m.bias, 0.0)

class network(torch.nn.Module):
    def __init__(self, opt, input_dim = 500, class_num = 79):
        super(network, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, class_num)

        self.apply(weights_init)

        self.train()

    def forward(self, x):
        z1 = F.relu(F.dropout(self.fc1(x)))
        z2 = F.relu(F.dropout(self.fc2(z1)))
        z3 = F.relu(F.dropout(self.fc3(z2)))
        output = self.fc4(z3)

        return output




