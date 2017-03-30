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
import math


class PAN17KLDivergenceModel(PAN17Classifier):

    # Constructor
    def __init__(self, classes=[], upper=False):
        """

        :param classes:
        """
        super(PAN17KLDivergenceModel, self).__init__()
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
    # end __init__

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
            prob[c] = self._classes_counts[c][token]
        # end for
        return prob
    # end word_probability

    # Compute the KL divergence between two distribution
    def _compute_kl_divergence(self, p, q):
        """

        :param p:
        :param q:
        :return:
        """
        count = Decimal(0.0)
        # For each words in distribution p
        for token in p.keys():
            if not self._upper:
                token = token.lower()
            # end if
            if p[token] > 0 and q[token] > 0:
                count += Decimal(Decimal(p[token]) * (Decimal(p[token]) / Decimal(q[token])).ln())
            # end if
        # end for
        return count
    # end _compute_KL_divergence

    # Evaluate unseen document
    def classify(self, tokens):
        """

        :param tokens:
        :return:
        """
        # Precision
        getcontext().prec = 256
        print("1")
        # Initialize prob
        total_tokens = 0.0
        doc_probs = dict()
        for token in self._collection_counts.keys():
            doc_probs[token] = Decimal(0.0)
        # end for
        for token in tokens:
            doc_probs[token] = Decimal(0.0)
        # end for
        print("2")
        # For each tokens
        for token in tokens:
            doc_probs[token] += Decimal(1.0)
            total_tokens += 1.0
        # end for
        print("3")
        # Calculate frequencies
        for token in doc_probs.keys():
            doc_probs[token] /= Decimal(total_tokens)
        # end for
        print("4")
        # Compute KL divergence for each classes
        kl_divs = dict()
        for c in self._classes_counts.keys():
            kl_divs[c] = self._compute_kl_divergence(doc_probs, self._collection_counts)
        # end
        print("5")
        mini = Decimal(10000000000.0)
        winner = ""
        # Get minimum divergence
        for c in self._classes_counts.keys():
            if kl_divs[c] < mini:
                mini = kl_divs[c]
                winner = c
            # end if
        # end for
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