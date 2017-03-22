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
from tools.PAN17Classifier import PAN17Classifier


class PAN17LanguageModel(PAN17Classifier):

    # Constructor
    def __init__(self, classes=[], upper=False):
        """

        :param classes:
        """
        super(PAN17LanguageModel, self).__init__()
        self._classes_counts = dict()
        self._classes_tokens_count = dict()
        self._collection_counts = dict()
        self._collection_tokens_count = 0
        for c in classes:
            self._classes_counts[c] = dict()
            self._classes_tokens_count[c] = 0
        # end for
        self._finalized = False
        self._upper = upper
        self._smoothing = None
    # end __init__

    # Set smoothing
    def set_smoothing(self, smoothing):
        self._smoothing = smoothing
    # end set_smoothing

    # Increment word
    def inc_word(self, c, key, n):
        """

        :param c:
        :param key:
        :param n:
        :return:
        """
        if not self._upper:
            key = key.lower()
        # end if
        self._classes_counts[c][key] += n
        self._classes_tokens_count[c] += n
        self._collection_counts[key] += n
        self._collection_tokens_count += n
    # end inc_word

    # Initialize token count
    def init_token_count(self, key):
        """

        :param key:
        :return:
        """
        if not self._upper:
            key = key.lower()
        # end if
        for c in self._classes_counts.keys():
            self._classes_counts[c][key] = 0.0
        # end for
        self._collection_counts[key] = 0.0
    # end init_token_count

    # Finalize model
    def finalize_model(self):
        """

        :return:
        """
        # For each registered words
        for c in self._classes_counts.keys():
            for token in self._classes_counts[c].keys():
                self._classes_counts[c][token] /= self._classes_tokens_count[c]
            # end for
        # end for
        # For each token
        for token in self._collection_counts.keys():
            self._collection_counts[token] /= self._collection_tokens_count
        # end for
        self._finalized = True
    # end finalize_model

    # Word probability
    def word_probability(self, token):
        """

        :param token:
        :return:
        """
        if not self._upper:
            token = token.lower()
        # end if
        prob = dict()
        # For each classes
        for c in self._classes_counts.keys():
            prob[c] = self._classes_counts[c][token] * 10.0
        # end for
        return prob
    # end word_probability

    # Evaluate unseen document
    def classify(self, tokens):
        """

        :param tokens:
        :return:
        """
        # Precision
        getcontext().prec = 256
        # Initialize prob
        prob = dict()
        for c in self._classes_counts.keys():
            prob[c] = Decimal(1.0)
        # end for
        # For each tokens
        for token in tokens:
            if not self._upper:
                token = token.lower()
            # end if
            token_prob = self.word_probability(token)
            """print("P : " + str(token_prob))
            print("C : " + str(self._collection_counts[token]))
            print("L : " + str(len(tokens)))"""
            if self._collection_counts[token] != 0:
                for c in prob.keys():
                    if self._smoothing is not None:
                        prob[c] *= Decimal(self._smoothing.smooth(token_prob[c], self._collection_counts[token],
                                                                  len(tokens)))
                    else:
                        prob[c] *= Decimal(token_prob[c])
                    # end if
                # end for
            # end if
            #print("R : " + str(prob))
        # end for
        # Get max prob
        max_prob = Decimal(0.0)
        winner = ""
        for c in prob.keys():
            if prob[c] > max_prob:
                winner = c
                max_prob = prob[c]
            # end if
        # end for
        #print("F : " + str(prob))
        #exit()
        return winner
    # end evaluate_doc

    # Get class model
    def get_class_model(self, c):
        class_model = []
        c_dict = self._classes_counts[c]
        for token in sorted(c_dict, key=c_dict.get, reverse=True):
            class_model += [(token, c_dict[token])]
        # end for
        return class_model
    # end get_class_model

# end PAN17ProbabilisticModel