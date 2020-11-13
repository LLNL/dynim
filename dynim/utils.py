# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import shutil
import datetime


# ------------------------------------------------------------------------------
def format_time(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


# ------------------------------------------------------------------------------
def take_backup(filename):

    if not os.path.isfile(filename):
        return

    extn = filename.split('.')[-1]
    bname = filename[:-len(extn)] + 'bak.' + extn
    shutil.move(filename, bname)


# ------------------------------------------------------------------------------
def write_history(filename, data, ts):

    if len(data) == 0:
        return

    ts = format_time(ts)
    with open(filename, 'a') as fp:
        for d in data:
            fp.write('{}, {}, {}\n'.format(ts, d.id, d.rank))

# ------------------------------------------------------------------------------
