# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import shutil
import time
import datetime
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def format_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


# ------------------------------------------------------------------------------
def backup_fname(filename):
    extn = filename.split('.')[-1]
    return filename[:-len(extn)] + 'bak.' + extn


def take_backup(filename):
    if os.path.isfile(filename):
        shutil.move(filename, backup_fname(filename))


# ------------------------------------------------------------------------------
def write_history(filename, event_type, data, state):

    if len(data) == 0:
        return

    ts = format_time()

    # write header if the file does not exist
    if not os.path.isfile(filename):
        with open(filename, 'a') as fp:
            fp.write('tstamp, event, id, rank, '
                     'nselected, ndiscarded, ncandidates, ncached\n')

    # now, write the data
    with open(filename, 'a') as fp:
        for d in data:
            if isinstance(d, str):
                _id, _rank = d, -1
            else:
                _id, _rank = d.id, d.rank

            fp.write('{}, {}, {}, {}, {}, {}, {}, {} \n'
                     .format(ts, event_type, _id, _rank,
                             state['nselected'], state['ndiscarded'],
                             state['ncandidates'], state['ncached']))


def read_history(filename):

    LOGGER.info('Reading history file ({})'.format(filename))
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None, encoding=None)

    dadd = np.array([d for d in data if d[1] == ' added'])
    dsel = np.array([d for d in data if d[1] == ' selected'])
    ddis = np.array([d for d in data if d[1] == ' discarded'])

    LOGGER.info('Found {} points (added={}, selected={}, discarded={})'
                .format(data.shape[0], dadd.shape[0], dsel.shape[0], ddis.shape[0]))
    assert data.shape[0] == dadd.shape[0] + dsel.shape[0] + ddis.shape[0]
    return dadd, dsel, ddis
# ------------------------------------------------------------------------------
