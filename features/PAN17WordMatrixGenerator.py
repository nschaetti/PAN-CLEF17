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

# Import packages
import numpy as np
import math
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


class PAN17WordMatrixGenerator(object):

    # Constructor
    def __init__(self, size, use_uppercase=False, bow=False, input_scaling=1.0):
        """

        :param size:
        """
        self._words = dict()
        self._size = size
        self._n_words = size * size
        self._word_index = dict()
        self._use_uppercase = use_uppercase
        self._bow = bow
        self._input_scaling = input_scaling
    # end __init__

    # Add tokens to the matrix
    def add_tokens(self, tokens):
        """

        :param tokens:
        :return:
        """
        for token in tokens:
            if not self._use_uppercase:
                token = token.lower()
            # end if
            #if len(self._words.keys()) < self._n_words:
            self._words[token] = True
            # end if
        # end for
    # end add_tokens

    # Finalize token index
    def finalize_token_index(self):
        """

        :return:
        """
        sorted_keys = sorted(self._words.keys())
        print(len(sorted_keys))
        index = 0
        for token in sorted_keys:
            if not self._use_uppercase:
                token = token.lower()
            # end if
            self._word_index[token] = index
            index += 1
        # end for
    # end finalize_token_index

    # Get token index in the matrix
    def _get_token_index(self, token):
        """

        :param token:
        :return:
        """
        if not self._use_uppercase:
            token = token.lower()
        # end if
        try:
            return self._word_index[token]
        except KeyError:
            return None
    # end _get_token_index

    # Index to matrix position
    def _index_to_position(self, index):
        div = float(index) / float(self._size)
        y = math.floor(div)
        x = (div - y) * float(self._size)
        return int(x), int(y)
    # end _index_to_position

    # Generate a matrix
    def generate_matrix(self, tokens):
        """

        :param tokens:
        :return:
        """
        #word_matrix = np.zeros((self._size, self._size))
        word_matrix = csr_matrix((self._size, self._size))
        for token in tokens:
            if not self._use_uppercase:
                token = token.lower()
            # end if
            index = self._get_token_index(token)
            if index is not None and index < self._n_words:
                x, y = self._index_to_position(index)
                if self._bow:
                    word_matrix[y, x] = 1.0
                else:
                    word_matrix[y, x] += 1.0
                # end if
            # end if
        # end for
        if not self._bow:
            return word_matrix / np.max(word_matrix) * self._input_scaling
        else:
            return word_matrix * self._input_scaling
        # end if
    # end generate_matrix

# end PAN17FeaturesMatrixGenerator