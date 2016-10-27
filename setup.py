from __future__ import (print_function, absolute_import)

from setuptools import setup, find_packages

long_description ="""Project Documentation: http://http://emaworkbench.readthedocs.io/"""

setup(
    name='ema_workbench',
    version='0.1.1dev',
    author='Jan Kwakkel',
    author_email='j.h.kwakkel@tudelft',
    packages=find_packages('./src', exclude=['test', 'test.*', 'examples']),
    package_dir={'':'./src'},
    url='https://github.com/quaquel/EMAworkbench',
    license='GNU General Public License',
    description='exploratory modelling in Python',
    long_description=long_description,
)
