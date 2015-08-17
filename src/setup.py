'''
Created on Aug 17, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from pip.req import parse_requirements
from setuptools import setup

install_reqs = parse_requirements('requirements.txt')

setup(name='EMAworkbench',
      version='0.5',
      description="support for exploratory modeling",
      long_description="",
      author='TODO',
      author_email='todo@example.org',
      license='TODO',
      packages=['analysis', 'connectors', 'expWorkbench'],
      zip_safe=False,
      install_requires=install_reqs,
      )