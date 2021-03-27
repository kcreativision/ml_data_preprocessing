from setuptools import setup

setup(
    name='data_curator',
    version='0.0.1',
    description='framework for data checks and preprocessing',
    packages=['data_checks', 'data_preprocessors'],
    install_requires=[] # add requirements.txt with scikit-learn
)
