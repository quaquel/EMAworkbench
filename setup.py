#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='ema_workbench',
    version='0.1dev',
    packages=find_packages(exclude=['ez_setup']),
    license='GNU General Public License',
    long_description=open('README.md').read(),
)