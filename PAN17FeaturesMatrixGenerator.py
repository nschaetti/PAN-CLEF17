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
import matplotlib.pyplot as plt


class PAN17FeaturesMatrixGenerator(object):

    # Constructor
    def __init__(self, letters="", punctuations="", upper_case=False):
        # Properties
        self._letters = letters
        self._punctuations = punctuations
        self._upper_case = upper_case
        self._end_letters_col = len(letters) if not upper_case else len(letters) * 2
        self._first_letters_col = self._end_letters_col + 1
        self._punctuations_col = self._first_letters_col + 1
        self._end_grams_col = self._punctuations_col + 1
        self._first_grams_col = self._end_grams_col + (len(letters) if not upper_case else len(letters) * 2)
    # end __init__

    # Compute matrix dimension
    def _compute_matrix_dimension(self, features_mapping):
        """

        :param features_mapping:
        :return:
        """
        # Row dimension
        n_row = len(self._letters)
        if self._upper_case:
            n_row *= 2
        # end
        n_row += 2

        # Grams
        n_col = 0
        if 'grams' in features_mapping:
            if self._upper_case:
                n_col += len(self._letters) * 2
            else:
                n_col += len(self._letters)
            # end if
        # end if

        # End grams
        if 'end_grams' in features_mapping:
            if self._upper_case:
                n_col += len(self._letters) * 2
            else:
                n_col += len(self._letters)
            # end if
        # end if

        # End letters
        if 'end_letters' in features_mapping:
            n_col += 1
        # end if

        # First grams
        if 'first_grams' in features_mapping:
            if self._upper_case:
                n_col += len(self._letters) * 2
            else:
                n_col += len(self._letters)
            # end if
        # end if

        # First letters
        if 'first_letters' in features_mapping:
            n_col += 1
        # end if

        # Punctuations
        if 'punctuations' in features_mapping:
            n_col += 1
        # end if

        return n_row, n_col
    # end _compute_matrix_dimension

    # Letters to row position
    def _letter_to_position(self, letter):
        try:
            pos = self._letters.index(letter.lower())
        except ValueError:
            print(letter.lower())
            exit()
        # end try
        if self._upper_case and letter.isupper():
            pos += len(self._letters)
        # end if
        return pos
    # end

    # Dictionnary sum
    def _dict_sum(self, dictionary):
        count = 0.0
        for key in dictionary.keys():
            count += dictionary[key]
        # end for
        return float(count)
    # end _dict_sum

    # Generate gram data
    def _generate_grams_data(self, features_mapping, mapping_index, col=0):
        # Matrix dimension
        n_row, n_col = self._compute_matrix_dimension(features_mapping)
        features_matrix = np.zeros((n_row, n_col))

        # Normalize
        divisor = self._dict_sum(features_mapping[mapping_index])

        # For each gram
        maxi = 0
        for gram in features_mapping[mapping_index].keys():
            count = features_mapping[mapping_index][gram]
            if count / divisor > maxi:
                maxi = count / divisor
            # end if
            if len(gram) == 1:
                if gram.isupper():
                    features_matrix[1, self._letter_to_position(gram.lower())] = count / divisor
                else:
                    features_matrix[0, self._letter_to_position(gram.lower())] = count / divisor
                # end if
            else:
                a = self._letter_to_position(gram[0])
                b = self._letter_to_position(gram[1])
                features_matrix[a+2, b+col] = count / divisor
            # end if
        # end for
        features_matrix /= maxi
        return features_matrix
    # end _generate_grams_data

    # Generate end letters data
    def _generate_letters_data(self, features_mapping, mapping_index, col_pos):
        # Matrix dimension
        n_row, n_col = self._compute_matrix_dimension(features_mapping)
        features_matrix = np.zeros((n_row, n_col))

        # Normalize
        divisor = self._dict_sum(features_mapping[mapping_index])

        # For each letters
        maxi = 0
        for letter in features_mapping[mapping_index].keys():
            count = features_mapping[mapping_index][letter]
            if count / divisor > maxi:
                maxi = count / divisor
            # end if
            a = self._letter_to_position(letter[0])
            features_matrix[a + 2, col_pos] = count / divisor
        # end for
        features_matrix /= maxi
        return features_matrix
    # end _generate_end_letters_data

    # Generate punctuations data
    def _generate_punctuations_data(self, features_mapping, mapping_index, col_pos):
        # Matrix dimension
        n_row, n_col = self._compute_matrix_dimension(features_mapping)
        features_matrix = np.zeros((n_row, n_col))

        # Normalize
        divisor = self._dict_sum(features_mapping[mapping_index])

        # For each punctuations
        maxi = 0
        for p in features_mapping[mapping_index].keys():
            count = features_mapping[mapping_index][p]
            if count / divisor > maxi:
                maxi = count / divisor
            # end if
            a = self._punctuations.index(p)
            features_matrix[a + 2, col_pos] = count / divisor
        # end for
        features_matrix /= maxi
        return features_matrix
    # end _generate_punctuations_data

    # Generate matrix
    def generate_matrix(self, features_mapping):
        # Matrix dimension
        n_row, n_col = self._compute_matrix_dimension(features_mapping)

        # Matrix
        features_matrix = np.zeros((n_row, n_col))

        # Generate grams data
        if 'grams' in features_mapping:
            features_matrix += self._generate_grams_data(features_mapping, 'grams')
        # end if

        # Generate end letters data
        if 'end_letters' in features_mapping:
            features_matrix += self._generate_letters_data(features_mapping, 'end_letters', self._end_letters_col)
        # end if

        # Generate end grams data
        if 'end_grams' in features_mapping:
            features_matrix += self._generate_grams_data(features_mapping, 'end_grams', self._end_grams_col)
        # end if

        # Generate first letters data
        if 'first_letters' in features_mapping:
            features_matrix += self._generate_letters_data(features_mapping, 'first_letters', self._first_letters_col)
        # end if

        # Generate punctuation data
        if 'punctuations' in features_mapping:
            features_matrix += self._generate_punctuations_data(features_mapping, 'punctuations',
                                                                self._punctuations_col)
        # end if

        # Generate first grams data
        if 'first_grams' in features_mapping:
            features_matrix += self._generate_grams_data(features_mapping, 'first_grams', self._first_grams_col)
        # end if

        return features_matrix
    # end generate_matrix

# end PAN17FeaturesMatrixGenerator