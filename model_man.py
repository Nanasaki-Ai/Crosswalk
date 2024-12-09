from __future__ import print_function
from net import *
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
             num_class=2):
        super().__init__()
        self.man = resnet50(class_num=num_class)


    def forward(self, x1, x2):
        x = self.man(x1, x2)
        return x