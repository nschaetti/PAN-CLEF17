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
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


class PAN17DiscreteSymbolGenerator(object):

    # Constructor
    def __init__(self, alphabet="", upper=False):
        self._alphabet = alphabet
        self._n_symbols = len(alphabet)
        self._symbols = dict()
        self._upper = upper
        self._generate_symbols()
    # end __init__

    # Generate discrete symbols
    def _generate_symbols(self):
        # For each letter in alphabet
        index = 0
        for l in self._alphabet:
            symbol = np.zeros(self._n_symbols)
            symbol[index] = 1.0
            self._symbols[l] = symbol
            index += 1
        # end for
    # end _generate_symbols

    # Letter to symbol
    def letter_to_symbol(self, letter):
        if self._upper:
            return self._symbols[letter]
        else:
            return self._symbols[letter.lower()]
        # end if
    # end letter_to_symbol

    # String to symbols list
    def string_to_symbols(self, s):
        symbols = []
        for l in s:
            symbols += [self.letter_to_symbol(l)]
        # end for
        return symbols
    # end string_to_symbols

    # Tokens to symbol list
    def tokens_to_symbols(self, tokens, sep=' '):
        symbols = []
        for token in tokens:
            symbols += self.string_to_symbols(token)
            symbols += [self.letter_to_symbol(sep)]
        # end for
        return symbols
    # end tokens_to_symbols

# end PAN17DiscreteSymbolGenerator