#!/usr/bin/env python3
from os.path import join, abspath, dirname
from setuptools import setup

with open(join(dirname(abspath(__file__)), 'requirements.txt')) as f:
    requirements = f.readlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

PLUGIN_ENTRY_POINT = 'ovos-padatious-pipeline-plugin=ovos_padatious.opm:PadatiousPipeline'


setup(
    name='ovos-padatious',
    version='0.4.8',  # Also change in ovos_padatious/__init__.py
    description='A neural network intent parser',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin',
    author='Matthew Scholefield',
    license='Apache-2.0',
    packages=[
        'ovos_padatious'
    ],
    install_requires=requirements,
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='intent-parser parser text text-processing',
    entry_points={'opm.pipeline': PLUGIN_ENTRY_POINT}
)
