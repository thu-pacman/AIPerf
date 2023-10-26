# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name='nnicli',
    version='1.0.0',
    packages=setuptools.find_packages(),

    python_requires='>=3.5',
    install_requires=[
        'requests'
    ],

    author='Microsoft NNI Team',
    author_email='nni@microsoft.com',
    description='nnicli for Neural Network Intelligence project',
    license='MIT',
    url='https://github.com/Microsoft/nni',
)
