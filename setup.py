from setuptools import setup, find_packages

setup(
    name             = 'medterm2vec',
    version          = '1.0',
    description      = 'medterm2vec ',
    long_description = open('README.md').read(),
    author           = 'Sanghyung Jin, Yourim Lee, Rae Woong Park',
    author_email     = 'jsh90612@gmail.com, urimeeee.e.gmail.com, rwpark99@gmail.com',
    url              = '',
    download_url     = '',
    install_requires = ['tensorflow', 'pandas', 'sklearn', 'scipy', 'numpy'],
    packages         = find_packages(exclude=["test*"]),
    keywords         = ['medical', 'sequence', 'embedding', 'machine learning', 'graph'],
    python_requires  = '>=3',
    package_data     = {},
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)