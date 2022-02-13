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
def files_exist(files):
    for f in files:
        if not os.path.isfile(f):
            return False
    return True


def files_to_restore(files):
    if files_exist(files):
        return files
    files = [backup_fname(f) for f in files]
    if files_exist(files):
        return files
    return []

# ------------------------------------------------------------------------------
def filter_list(data_to_filter, filter_if_present_in_list):
    assert isinstance(data_to_filter, list)
    assert isinstance(filter_if_present_in_list, (list, np.ndarray))

    if len(data_to_filter) == 0 or len(filter_if_present_in_list) == 0:
        return data_to_filter, []

    # since filter_if_present_in_list is likely very large,
    # let's use numpy to find the ones that are already included
    ids_in_data = np.array([_.id for _ in data_to_filter])
    ids_in_list = np.array([_.id for _ in filter_if_present_in_list])
    is_present = np.intersect1d(ids_in_data, ids_in_list)

    # no intersection
    if is_present.shape[0] == 0:
        return data_to_filter, []

    # slower list comprehension to find the ones not present
    is_not_present = [_ for _ in data_to_filter if _.id not in is_present]
    return is_not_present, is_present.tolist()


# new attempt for faster filtering, but requires separate list of ids
def filter_list_v1(data_to_filter, filter_if_present_in_list):

    assert isinstance(data_to_filter, list)
    assert isinstance(filter_if_present_in_list, (list, np.ndarray))

    if len(data_to_filter) == 0 or len(filter_if_present_in_list) == 0:
        return data_to_filter, []

    _data = np.unique(data_to_filter)
    _new_data = np.setdiff1d(_data, filter_if_present_in_list, assume_unique = True)

    if len(_data) == len(new_data):
        return _new_data.tolist(), []

    # else
    _old_data = np.setdiff1d(_data, new_data, assume_unique = True)
    return _new_data.tolist(), _old_data.tolist()

# ------------------------------------------------------------------------------
# @TODO fix this added just to make code work
def write_history_ids(filename, event_type, data, state):

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
            fp.write('{}, {}, {}, {}, {}, {}, {}, {} \n'
                     .format(ts, event_type, d, state, "??", "??", "??", "??"))


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
