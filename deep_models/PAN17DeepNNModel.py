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
    def __init__(self, model, classes=[], cuda=False, lr=0.01, momentum=0.5, log_interval=10, seed=1):
        """

        :param classes:
        """
        super(PAN17DeepNNModel, self).__init__()
        self._classes = classes
        self._log_interval = log_interval
        #self._model = PAN17ConvNet()
        self._model = model
        self._cuda = cuda
        if cuda:
            self._model.cuda()
        self._model.double()
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._train_loader = None
        self._kwargs = {'num_workers': 1, 'pin_memory': True} if self._cuda else {}
    # end __init__

    # Class to int
    def _class_to_int(self, c):
        """

        :param c:
        :return:
        """
        return self._classes.index(c)
    # end _class_to_int

    # Int to class
    def _int_to_class(self, i):
        """

        :param i:
        :return:
        """
        return self._classes[i]
    # end _int_to_class

    # Numpy matrix to Tensor
    def matrix_to_tensor(self, m):
        #m = m.toarray()
        h = int(m.shape[0])
        w = int(m.shape[1])
        x = torch.DoubleTensor(1, h, w)
        for i in range(h):
            for j in range(w):
                x[0, i, j] = m[i, j]
            # end for
        # end for
        return x
    # end _matrix_to_tensor

    # To Torch data set
    def to_torch_data_set(self, matrices, truths):
        """

        :param matrices:
        :param truths:
        :return:
        """
        result = []
        for index, (m) in enumerate(matrices):
            #print("%d on %d" % (index, len(matrices)))
            truth = truths[index]
            result += [(self.matrix_to_tensor(m), self._class_to_int(truth))]
            #print("%d on %d" % (index, len(matrices)))
        # end for
        return result
    # end to_torch_data_set

    # Train
    def train(self, epoch, data_set, batch_size=64):
        """
        Train
        :param epoch:
        :param matrices:
        :param truths:
        :param batch_size:
        :return:
        """
        # To Torch data set
        train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, **self._kwargs)

        # Put model in train mode
        self._model.train()

        # Loss
        training_loss = 0

        # For each batch
        for batch_idx, (data, target) in enumerate(train_loader):
            # Create variables
            if self._cuda:
                data, target = data.cuda(), target.cuda()
            # end if
            data, target = Variable(data), Variable(target)

            # ??
            self._optimizer.zero_grad()

            # Get model output
            output = self._model(data)

            # Get loss
            loss = F.nll_loss(output, target)
            training_loss += loss.data[0]

            # Apply difference backward
            loss.backward()

            # ??
            self._optimizer.step()
        # end for

        # Loss function already averages over batch size
        training_loss = training_loss
        training_loss /= len(train_loader)

        # Print
        print("Iteration {}: Training Loss: {:.4f}".format(epoch, training_loss))
    # end train

    # Evaluate unseen document
    def test(self, epoch, data_set, batch_size=64):
        """

        :param tokens:
        :return:
        """
        # ??
        self._model.eval()

        # Loss and correct
        test_loss = 0
        correct = 0

        # To Torch data set
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, **self._kwargs)

        # For each batch
        for data, target in test_loader:
            # Variables
            if self._cuda:
                data, target = data.cuda(), target.cuda()
            # end if
            data, target = Variable(data), Variable(target)

            # Model's output
            output = self._model(data)

            # Loss
            test_loss += F.nll_loss(output, target).data[0]

            # Get the index if the max log-probability
            pred = output.data.max(1)[1]

            # Add if correct
            correct += pred.eq(target.data).cpu().sum()
        # end for

        # Loss function already averages over batch size
        test_loss = test_loss
        test_loss /= len(test_loader)

        # Print informations
        print("Iteration {}: Average test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100.0 * float(correct) / float(len(test_loader.dataset))
        ))
    # end evaluate_doc

# end PAN17ProbabilisticModel
