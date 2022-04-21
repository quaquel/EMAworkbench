import io
import os
import re

from setuptools import setup


def read(path, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(
        r"""^__version__ = ['"]([^'"]*)['"]""", version_file, re.M
    )
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


example_data_files = package_files("ema_workbench/examples", "data")
example_data_files = ["".join(["examples/", entry]) for entry in example_data_files]

example_model_files = package_files("ema_workbench/examples", "models")
example_model_files = ["".join(["examples/", entry]) for entry in example_model_files]

name = "ema_workbench"
pjoin = os.path.join
here = os.path.abspath(os.path.dirname(__file__))
pkg_root = pjoin(here, name)

packages = []
for d, _, _ in os.walk(pkg_root):
    if os.path.exists(pjoin(d, "__init__.py")):
        packages.append(d[len(here) + 1 :].replace(os.path.sep, "."))

VERSION = version("ema_workbench/__init__.py")
LONG_DESCRIPTION = "Project Documentation: https://emaworkbench.readthedocs.io/"
EXAMPLE_DATA = example_data_files + example_model_files
PACKAGES = packages


setup(
    name="ema_workbench",
    version=VERSION,
    description="exploratory modelling in Python",
    long_description=LONG_DESCRIPTION,
    author="Jan Kwakkel",
    author_email="j.h.kwakkel@tudelft.nl",
    packages=PACKAGES,
    package_data={"ema_workbench": EXAMPLE_DATA},
    url="https://github.com/quaquel/EMAworkbench",
    license="BSD 3-Clause",
    platforms="Linux, Mac OS X, Windows",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
