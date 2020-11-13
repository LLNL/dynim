# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

from setuptools import setup, find_packages

__version_info__ = ('0', '1', '0')
__version__ = '.'.join(__version_info__)

# ------------------------------------------------------------------------------
setup(
    name='dynim',
    version=__version__,
    license='MIT',
    description='Dynamic Importance Sampling',
    url='https://github.com/LLNL/dynim',
    author=['Harsh Bhatia', 'Joseph Y. Moon'],
    author_email=['hbhatia@llnl.gov', 'moon15@llnl.gov'],
    keywords='',
    packages=find_packages(),
    install_requires=['numpy', 'faiss', 'pyyaml'],
)

# ------------------------------------------------------------------------------
