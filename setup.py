from setuptools import setup, find_packages

from gp2.runner import __version__

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
    extra_require={
        'gp2': [
            'python==3.9.7',
            'typing-extensions==3.7.4.3',
            'absl-py==0.15.0',
            'astor==0.8.1',
            'astunparse==1.6.3',
            'cachetools==5.0.0',
            'catboost==1.1.1',
            'category-encoders==2.6.0',
            'charset-normalizer==2.0.12',
            'combo==0.1.3',
            'contextlib2==21.6.0',
            'eif==1.0.2',
            'flatbuffers==1.12',
            'gast==0.4.0',
            'grpcio==1.34.1',
            'h5py==3.1.0',
            'importlib-metadata==4.11.2',
            'json-tricks==3.16.1',
            'keras-nightly==2.5.0.dev2021032900',
            'keras-preprocessing==1.1.2',
            'keras-unet-collection==0.1.13',
            'lightgbm==3.3.5',
            'markdown==3.3.6',
            'mlxtend==0.21.0',
            'numpy==1.19.5',
            'oauthlib==3.2.0',
            'opencv-python==4.7',
            'opt-einsum==3.3.0',
            'plotly==5.13.0',
            'prettierplot==0.1.2',
            'prettytable==3.6.0',
            'protobuf==3.19.4',
            'psutil==5.9.1',
            'pyasn1-modules==0.2.8',
            'pygments==2.15.1',
            'pylibjpeg==1.0.0',
            'pylibjpeg-libjpeg==1.0.0',
            'pythonwebhdfs==0.2.3',
            'requests==2.27.1',
            'responses==0.22.0',
            'rsa==4.8',
            'scikit-learn==1.2.1',
            'seaborn==0.12.2',
            'tensorboard==2.8.0',
            'tensorboard-data-server==0.6.1',
            'tensorboard-plugin-wit==1.8.1',
            'tensorflow==2.5.0',
            'tensorflow-estimator==2.5.0',
            'keras-unet==0.1.2'


        ]
    }

)
