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

from PAN17LanguageModelSmoothing import PAN17LanguageModelSmoothing


class PAN17JelinekMercerSmoothing(PAN17LanguageModelSmoothing):

    # Constructor
    def __init__(self, l):
        super(PAN17JelinekMercerSmoothing, self).__init__()
        self._lambda = l
    # end __init__

    # Smooth function
    def smooth(self, doc_prob, col_prob):
        return (1.0 - self._lambda) * doc_prob + self._lambda * col_prob
    # end smooth

# end PAN17JelinekMercerSmoothing