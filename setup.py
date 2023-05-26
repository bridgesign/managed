#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='tensor-managed',
        version='0.1',
        description='Automated GPU Allocation for PyTorch',
        author='Rohan Patil',
        author_email="eyeoeternity@gmail.com",
        url="https://github.com/bridgesign/managed",
        packages=find_packages(exclude=('tests',)),
        license='MIT',
        install_requires=[
            'torch',
        ],
    )