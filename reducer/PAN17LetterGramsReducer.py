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
from pySpeeches.mapreduce.PySpeechesMapReducer import *


# An Author
class PAN17LetterGramsReducer(PySpeechesMapReducer):

    # Constructor
    def __init__(self, letters="abcdefghijflmnopqrstuvwxyz", punctuations=".,;:!?", n_gram=2,
                 add_punctuation=False, add_end_letters=False, add_end_grams=False, add_first_letters=False,
                 add_first_grams=False, upper_case=False):
        super(PAN17LetterGramsReducer, self).__init__()
        self._letters = letters
        self._punctuations = punctuations
        self._n_gram = n_gram
        self._add_punctuation = add_punctuation
        self._add_end_letter = add_end_letters
        self._add_end_grams = add_end_grams
        self._add_first_letters = add_first_letters
        self._add_first_grams = add_first_grams
        self._upper_case = upper_case
    # end

    # Get token grams
    def get_token_grams(self, token):
        result = dict()
        if not self._upper_case:
            token = token.lower()
        # end if
        if len(token) > 1:
            for i in np.arange(1, len(token)):
                if token[i-1:i+1] not in result:
                    result[token[i-1:i+1]] = 1
                else:
                    result[token[i - 1:i + 1]] += 1
                # end if
            # end for
        elif token[0] in self._letters or token[0] in self._letters.upper():
            result[token[0]] = 1
        # end if
        return result
    # end get_token_grams

    # Map punctuations
    def map_punctuations(self, doc):
        """

        :param doc:
        :return:
        """
        result = dict()
        for token in doc:
            if len(token) == 1 and token[0] in self._punctuations:
                if token[0] not in result:
                    result[token[0]] = 1
                else:
                    result[token[0]] += 1
                # end if
            # end if
        # end for
        return result
    # end map_punctuations

    # Map grams
    def map_grams(self, doc):
        """

        :param doc:
        :return:
        """
        result = dict()
        for token in doc:
            grams = self.get_token_grams(token)
            for gram in grams.keys():
                count = grams[gram]
                if gram not in result:
                    result[gram] = 1
                else:
                    result[gram] += count
                # end if
            # end for
        # end for
        return result
    # end map_grams

    # Map first letters
    def map_first_letters(self, doc):
        """
        Map first letters
        :param doc:
        :return:
        """
        result = dict()
        for token in doc:
            if not self._upper_case:
                token = token.lower()
            # end if
            if len(token) > 1 or token[0] not in self._punctuations:
                if token[0] not in result:
                    result[token[0]] = 1
                else:
                    result[token[0]] += 1
                # end if
            # end if
        # end for
        return result
    # end map_first_letters

    # Map first grams
    def map_first_grams(self, doc):
        """
        Map first grams
        :param doc: The document to map.
        :return: The map
        """
        result = dict()
        gram = ""
        for token in doc:
            if not self._upper_case:
                token = token.lower()
            # end if
            if len(token) == 1 and token[0] not in self._punctuations:
                gram = token[0]
            elif len(token) > 1:
                gram = token[0] + token[1]
            # end if
            if gram != "":
                if gram not in result:
                    result[gram] = 1
                else:
                    result[gram] += 1
                # end if
            # end if
        # end for
        return result
    # end map_first_grams

    # Map end letters
    def map_end_letters(self, doc):
        """
        Map end letters.
        :param doc: The document to maps.
        :return: Maps of the end letters.
        """
        result = dict()
        for token in doc:
            if not self._upper_case:
                token = token.lower()
            # end if
            if len(token) > 1 or token[-1] not in self._punctuations:
                if token[-1] not in result:
                    result[token[-1]] = 1
                else:
                    result[token[-1]] += 1
                # end if
            # end if
        # end for
        return result
    # end map_end_letters

    # Map end grams
    def map_end_grams(self, doc):
        """

        :param doc:
        :return:
        """
        result = dict()
        gram = ""
        for token in doc:
            if not self._upper_case:
                token = token.lower()
            # end if
            if len(token) == 1 and token[0] not in self._punctuations:
                gram = token[0]
            elif len(token) > 1:
                gram = token[-2] + token[-1]
            # end if
            if gram != "":
                if gram not in result:
                    result[gram] = 1
                else:
                    result[gram] += 1
                # end if
            # end if
        # end for
        return result
    # end map_end_grams

    # Map the document
    def map(self, doc):
        """

        :param doc:
        :return:
        """
        result = dict()

        # Grams
        result['grams'] = self.map_grams(doc)

        # Punctuations
        if self._add_punctuation:
            result['punctuations'] = self.map_punctuations(doc)
        # end if

        # First letter
        if self._add_first_letters:
            result['first_letters'] = self.map_first_letters(doc)
        # end if

        # First grams
        if self._add_first_grams:
            result['first_grams'] = self.map_first_grams(doc)
        # end if

        # End letter
        if self._add_end_letter:
            result['end_letters'] = self.map_end_letters(doc)
        # end if

        # End grams
        if self._add_end_grams:
            result['end_grams'] = self.map_end_grams(doc)
        # end if

        return result
    # end map

    # Merge two mapping
    def _merge_mappings(self, map1, map2):
        result = dict()
        for key in map1:
            if key in map2:
                result[key] = map1[key] + map2[key]
            else:
                result[key] = map1[key]
            # end if
        # end for
        for key in map2:
            if key not in map1:
                result[key] = map2[key]
            # end if
        # end for
        return result
    # end _merge_mappings

    # Merge two mapping
    def _merge_two_mappings(self, map1, map2):
        if len(map1) == 0:
            return map2
        # end if
        if len(map2) == 0:
            return map1
        # end if

        # Result
        result = dict()

        # For reach key
        for key in map1:
            if key in map2.keys():
                result[key] = self._merge_mappings(map1[key], map2[key])
            # end if
        # end for
        return result
    # end _merge_two_mappings

    # Reduce the data
    def reduce(self, mapping_list):
        final_mapping = dict()
        # For each mapping
        for mapping in mapping_list:
            final_mapping = self._merge_two_mappings(final_mapping, mapping)
        # end for
        return final_mapping
    # end reduce

# end PAN17LetterGramsReducer
