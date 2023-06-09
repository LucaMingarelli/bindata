"""  Created on 12/03/2023::
------------- setup.py -------------

**Authors**: L. Mingarelli
"""

import setuptools
import bindata as bnd

with open("README.md", 'r') as f:
    long_description = f.read()

with open("bindata/requirements.txt") as f:
    install_requirements = f.read().splitlines()


setuptools.setup(
    name="bindata",
    version=bnd.__version__,
    author=bnd.__author__,
    author_email=bnd.__email__,
    description=bnd.__about__,
    url=bnd.__url__,
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['bindata', 'bindata.tests', 'bindata.res'],
    package_data={'':  ['../bindata/res/*']},
    install_requires=install_requirements,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.6',
)

