"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer


setup(
    name='pyiron_continuum',
    version=versioneer.get_version(),
    description='Repository for user-generated plugins to the pyiron IDE.',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/pyiron_continuum',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='huber@mpie.de',
    license='BSD',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[
        'matplotlib==3.8.4',
        'numpy==1.26.4',
        'pyiron_base==0.9.1',
        'pyiron_snippets==0.1.0',
        'scipy==1.13.0',
        'sympy==1.12'
    ],
    extras_require={
        'fenics': [
            'fenics==2019.1.0',
            'mshr==2019.1.0',
        ],
        'schroedinger': ['k3d==2.16.1']
    },
    cmdclass=versioneer.get_cmdclass(),
    
)
