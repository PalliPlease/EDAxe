from setuptools import setup, find_packages

setup(
    name='EDAxe',
    version='0.1.0',
    description='Automated exploratory data analysis and preprocessing toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Parv Joshi',
    author_email='parvjoshi2k4@gmail.com',
    url='https://github.com/PalliPlease/EDAxe',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas>=1.5',
        'numpy>=1.21',
        'scikit-learn>=1.3',
        'matplotlib>=3.5',
        'seaborn>=0.11'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # um idk
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.7',
)
