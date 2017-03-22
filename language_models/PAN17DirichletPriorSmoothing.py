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


class PAN17DirichletPriorSmoothing(PAN17LanguageModelSmoothing):

    # Constructor
    def __init__(self, mu):
        super(PAN17DirichletPriorSmoothing, self).__init__()
        self._mu = mu
    # end __init__

    # Smooth function
    def smooth(self, doc_prob, col_prob, doc_length):
        return (float(doc_length) / (float(doc_length) + float(self._mu))) * doc_prob + \
               (float(self._mu) / (float(self._mu) + float(doc_length))) * col_prob
    # end smooth

# end PAN17DirichletPriorSmoothing