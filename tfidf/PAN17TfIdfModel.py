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
from numpy import linalg as LA
import math
from decimal import *
from tools.PAN17Classifier import PAN17Classifier


class PAN17TfIdfModel(PAN17Classifier):

    # Constructor
    def __init__(self, classes=[], upper=False, use_punct=True, punc = ""):
        """

        :param classes:
        :param upper:
        """
        super(PAN17TfIdfModel, self).__init__()
        self._classes_counts = dict()
        self._classes_token_count = dict()
        self._collection_counts = dict()
        self._classes_vectors = dict()
        self._classes_frequency = dict()
        for c in classes:
            self._classes_counts[c] = dict()
            self._classes_token_count[c] = 0.0
        # end for
        self._upper = upper
        self._token_position = dict()
        self._n_classes = float(len(classes))
        self._use_punct = use_punct
        self._punc = punc
    # end __init__

    def _filter_token(self, token):
        if not self._upper:
            token = token.lower()
        # end if
        if not self._use_punct and token in self._punc:
            return None
        # end if
        return token
    # end _filter_token

    # Initialize token count
    def init_token_count(self, key):
        """

        :param key:
        :return:
        """
        key = self._filter_token(key)
        if key is not None:
            for c in self._classes_counts.keys():
                self._classes_counts[c][key] = 0.0
            # end for
            self._collection_counts[key] = 0.0
            self._classes_frequency[key] = 0.0
        # end if
    # end init_token_count

    # Train the model
    def train(self, tokens, c):
        """

        :param tokens:
        :param c:
        :return:
        """
        # For each tokens
        for token in tokens:
            token = self._filter_token(token)
            if token is not None:
                self._classes_counts[c][token] += 1.0
                self._collection_counts[token] += 1.0
                self._classes_token_count[c] += 1.0
            # end if
        # end for
    # end train

    # Finalize
    def finalize(self):
        # Position of each token
        i = 0
        for token in sorted(self._collection_counts.keys()):
            self._token_position[token] = i
            i += 1
        # end for
        # Compute classes frequency
        for token in self._collection_counts.keys():
            count = 0.0
            for c in self._classes_counts.keys():
                if self._classes_counts[c][token] > 0:
                    count += 1.0
                # end if
            # end for
            self._classes_frequency[token] = count
            # end for
        # end if
        # For each classes
        for c in self._classes_counts.keys():
            c_vector = np.zeros(len(self._classes_counts[c].keys()), dtype='float64')
            for token in self._collection_counts.keys():
                index = self._token_position[token]
                c_vector[index] = self._classes_counts[c][token]
            # end for
            c_vector /= float(self._classes_token_count[c])
            for token in self._collection_counts.keys():
                index = self._token_position[token]
                if self._classes_frequency[token] > 0:
                    c_vector[index] *= math.log(self._n_classes / self._classes_frequency[token])
                # end if
            # end for
            self._classes_vectors[c] = c_vector
        # end for
    # end finalize

    # Cosinus similarity
    def _cosinus_similarity(self, a, b):
        return np.dot(a, b) / (LA.norm(a) * LA.norm(b))
    # end _cosinus_similarity

    # Evaluate unseen document
    def classify(self, tokens):
        """

        :param tokens:
        :return:
        """
        d_vector = np.zeros(len(self._collection_counts.keys()), dtype='float64')
        for token in tokens:
            token = self._filter_token(token)
            if token is not None:
                try:
                    index = self._token_position[token]
                    d_vector[index] += 1.0
                except KeyError:
                    pass
                # end try
            # end if
        # end for

        # Normalize vector
        d_vector /= float(len(tokens))

        # For each classes
        similarity = np.zeros(len(self._classes_counts.keys()))
        index = 0
        for c in self._classes_counts.keys():
            similarity[index] = self._cosinus_similarity(self._classes_vectors[c], d_vector)
            index += 1
        # end for
        #print(distances)
        return self._classes_counts.keys()[np.argmax(similarity)]
    # end evaluate_doc

    # Get class vectors
    def get_vectors(self):
        vectors = dict()
        for c in self._classes_counts.keys():
            vectors[c] = self._classes_vectors[c]
        # end for
        return vectors
    # end get_vectors

# end PAN17VectorModel