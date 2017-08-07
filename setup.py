from __future__ import (print_function, absolute_import)

import io
import os
import re

from setuptools import setup, find_packages


def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def package_files(root, package_data_dir):
    paths = []
    
    directory = os.path.join(root, package_data_dir)
    
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            full_path = os.path.join(path, filename)
            
            paths.append(os.path.relpath(full_path, root))
    return paths

example_data_files = package_files('src/examples', 'data')
example_model_files = package_files('src/examples', 'models')

VERSION = version('src/ema_workbench/__init__.py')
LONG_DESCRIPTION ="""Project Documentation: https://emaworkbench.readthedocs.io/"""
EXAMPLE_DATA = example_data_files + example_model_files

setup(
    name='ema_workbench',
    version=VERSION,
    author='Jan Kwakkel',
    author_email='j.h.kwakkel@tudelft',
    packages=find_packages('./src', exclude=['test', 'test.*']),
    package_dir={'':'./src'},
    package_data = {'examples': EXAMPLE_DATA},
    url='https://github.com/quaquel/EMAworkbench',
    license='BSD 3-Clause',
    description='exploratory modelling in Python',
    long_description=LONG_DESCRIPTION,
)
