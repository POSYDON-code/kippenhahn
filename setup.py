#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Tassos Fragos (2023)
#
# This file is part of the kippenhahn python package.
#
# kippenhahn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kippenhahn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kippenhahn.  If not, see <http://www.gnu.org/licenses/>.

"""Setup the kippenhahn package
"""

from __future__ import print_function

import glob
import os.path
import sys

from setuptools import (setup, find_packages)

cmdclass = {}

# -- versioning ---------------------------------------------------------------

import versioneer
__version__ = versioneer.get_version()
cmdclass.update(versioneer.get_cmdclass())

# -- documentation ------------------------------------------------------------

# # import sphinx commands
# try:
#     from sphinx.setup_command import BuildDoc
# except ImportError:
#     pass
# else:
#     cmdclass['build_sphinx'] = BuildDoc

# read description
with open('README.md', 'rb') as f:
    longdesc = f.read().decode().strip()

# -- dependencies -------------------------------------------------------------

if 'test' in sys.argv:
    setup_requires = [
        'setuptools',
        'pytest-runner',
    ]
else:
    setup_requires = []

# These pretty common requirement are commented out. Various syntax types
# are all used in the example below for specifying specific version of the
# packages that are compatbile with your software.
install_requires = [
    # 'numpy >= 1.12.0',
    # #'pyblast @ https://github.com/CIERA-Northwestern/pyblast/tarball/master',
    # 'scipy >= 0.12.1',
    # 'matplotlib >= 1.2.0, != 2.1.0, != 2.1.1',
    # 'astropy >= 1.1.1, < 3.0.0 ; python_version < \'3\'',
    # #'astropy >= 1.1.1 ; python_version >= \'3\'',
    # #'configparser',
    # #'pandas >= 0.24',
]

tests_require = [
    "pytest >= 3.3.0",
    "pytest-cov >= 2.4.0",
]

# For documenation
extras_require = {
    'doc': [
        'matplotlib',
        'ipython',
        'sphinx',
        'numpydoc',
        'sphinx_rtd_theme',
        'sphinxcontrib_programoutput',
    ],
    'ml': ['theano'],
}

# -- run setup ----------------------------------------------------------------

packagenames = find_packages()

# Executables go in a folder called bin
scripts = glob.glob(os.path.join('bin', '*'))

PACKAGENAME = 'kippenhahn'
DISTNAME = 'kippenhahn'
AUTHOR = 'Tassos Fragos'
AUTHOR_EMAIL = 'anastasios.fragkos@unige.ch'
LICENSE = 'GPLv3+'
DESCRIPTION = 'An awesome Kippenhahn diagram generator'
GITHUBURL = 'https://github.com/POSYDON-code/kippenhahn'

setup(name=DISTNAME,
      provides=[PACKAGENAME],
      version=__version__,
      description=DESCRIPTION,
      long_description=longdesc,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=packagenames,
      include_package_data=True,
      cmdclass=cmdclass,
      url=GITHUBURL,
      scripts=scripts,
      setup_requires=setup_requires,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      python_requires='>3.5, <4',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
      ],
)
