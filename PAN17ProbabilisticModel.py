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


class PAN17ProbabilisticModel(object):

    # Constructor
    def __init__(self, classes=[], upper=False):
        """

        :param classes:
        """
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

    # Evaluate unseen document
    def evaluate_doc(self, tokens):
        """

        :param tokens:
        :return:
        """
        # Initialize prob
        prob = dict()
        for c in self._classes_counts.keys():
            prob[c] = 1.0
        # end for
        # For each tokens
        for token in tokens:
            if not self._upper:
                token = token.lower()
            # end if
            token_prob = self.word_probability(token)
            for c in prob.keys():
                prob[c] *= token_prob[c]
            # end for
        # end for
        return prob
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