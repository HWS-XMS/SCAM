"""
Setup script for SCAM (Side Channel Analysis Measurements) package.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'scam', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

# Read README for long description
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='scam',
    version=get_version(),
    author='SCAM Contributors',
    author_email='info@example.com',
    description='SCAM - Side Channel Analysis Measurements',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/example/scam',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'h5py>=3.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
        ],
        'examples': [
            'matplotlib>=3.0',
            'pycryptodome>=3.0',
        ],
    },
    entry_points={
        'console_scripts': [
            # Add command-line tools here if needed
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/example/scam/issues',
        'Source': 'https://github.com/example/scam',
        'Documentation': 'https://scam.readthedocs.io/',
    },
)