from __future__ import (print_function, absolute_import)

import io
import os
import re

from setuptools import setup


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

example_data_files = package_files('examples', 'data')
example_model_files = package_files('examples', 'models')
java_files = package_files('ema_workbench/connectors', 'java')
java_files = [''.join(['connectors/', entry]) for entry in java_files]

# print(example_data_files)
# print(java_files)

VERSION = version('ema_workbench/__init__.py')
LONG_DESCRIPTION ="""Project Documentation: https://emaworkbench.readthedocs.io/"""
EXAMPLE_DATA = example_data_files + example_model_files
JAVA = java_files
PACKAGES = ['ema_workbench', 
            'ema_workbench.analysis', 
            'ema_workbench.connectors', 
            'ema_workbench.em_framework', 
            'ema_workbench.util', 
            'ema_workbench.analysis.cluster_util',
            'examples']

setup(
    name='ema_workbench',
    version=VERSION,
    author='Jan Kwakkel',
    author_email='j.h.kwakkel@tudelft',
    packages=PACKAGES,
    package_data = {'examples': EXAMPLE_DATA, 
                    'ema_workbench':JAVA},
    url='https://github.com/quaquel/EMAworkbench',
    license='BSD 3-Clause',
    description='exploratory modelling in Python',
    long_description=LONG_DESCRIPTION,
)
