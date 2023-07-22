from setuptools import setup, find_packages

from gp2 import __version__

setup(
    name='gp2',
    version=__version__,
    url='https://github.com/RyanZurrin/CS410-GP2',
    description='A framework for tuning segmentation classifiers and '
                'discriminators',
    author='mpsych lab',
    packages=find_packages(),
    classifiers=[
        'Environment :: GPU :: NVIDIA CUDA :: 11.3',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    install_requires=[
        'tensorflow>=2.5.0',
        'matplotlib>=3.4.1',
        'keras_unet_collection>=0.1.1',
        'keras-unet>=0.1.2',
        'numpy>=1.19.5',
    ],

)
