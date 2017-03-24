# -*- coding: utf-8 -*-
#
# File : core/downloader/PySpeechesConfig.py
# Description : .
# Date : 20th of February 2017
#
# This file is part of pySpeeches.  pySpeeches is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

import numpy as np
from decimal import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tools.PAN17Classifier import PAN17Classifier
from PAN17ConvNet import PAN17ConvNet


class PAN17DeepNNModel(PAN17Classifier):

    # Constructor
    def __init__(self, classes=[], cuda=False, lr=0.01, momentum=0.5):
        """

        :param classes:
        """
        super(PAN17DeepNNModel, self).__init__()
        self._model = PAN17ConvNet()
        if cuda:
            self._model.cuda()
        self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._kwargs = {'num_workers' : 1, 'pin_memory': True} if cuda else {}
        self._train_loader = None
    # end __init__

    # Train
    def train(self, epoch, batch_size=64):
        """
        Train the model.
        :param epoch:
        :return:
        """
        self._train_loader = torch.utils.data.DataLoader(epoch, batch_size=batch_size, shuffle=False, **self._kwargs)
        self._model.train()
        for batch_idx
    # end train

    # Evaluate unseen document
    def classify(self, epoch):
        """

        :param tokens:
        :return:
        """
        self._model.eval()
        test_loss = 0
        correct = 0
        for data, target in
    # end evaluate_doc

# end PAN17ProbabilisticModel