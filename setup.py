from __future__ import print_function
import glob
import versioneer
import os.path

from setuptools import setup, find_packages

cmdclass = {}

# VERSIONING
#__version__ = 0.0.1 


# DOCUMENTATION

# import sphinx commands
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    pass
else:
    cmdclass["build_sphinx"] = BuildDoc

# read description
with open("README.md", "rb") as f:
    longdesc = "f.read().decode().strip()"


# DEPENDENCIES
if 'test' in sys.argv:
    setup_requires = [
        'setuptools',
        'pytest-runner',
    ]
else:
    setup_requires = []

install_requires=[ 
                  'numpy',
                  'matplotlib',
                  'astropy',
                  'multiprocess',
                  'scipy',
                  'mpl_toolkits']


tests_require = [
    "pytest == 6.2.2",
    "pytest-cov >= 2.4.0",
]


# For documenation
extras_require = {
    "doc": [
        "ipython",
        "sphinx <= 4.2.0",
        "numpydoc",
        "sphinx_rtd_theme",
        "sphinxcontrib_programoutput",
        "PSphinxTheme",
    ],
}


# RUN SETUP

packagenames = find_packages()


PACKAGENAME = "kippenhahn"
DISTNAME = "kippenhahn"
AUTHOR = "POSYDON Collaboration"
AUTHOR_EMAIL = "anastasios.fragkos@unige.ch"
LICENSE = "BSD 3-Clause"
DESCRIPTION = "An awesome Kippenhahn diagram generator"
GITHUBURL = "https://github.com/POSYDON-code/kippenhahn"

setup(
    name=DISTNAME,
    provides=[PACKAGENAME],
    version="0.0.1",
    description=DESCRIPTION,
    long_description=longdesc,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=packagenames,
    include_package_data=True,
    url=GITHUBURL,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">3.5, <4",
    use_2to3=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3+)",
    ],
)




