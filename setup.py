from setuptools import setup, find_packages

import os
from os.path import join

def configuration():
    print "Found packages: %s" % find_packages()

    params = {
        'name': 'tensorflow-classifier',
        'version': "0.1-SNAPSHOT",
        'url': 'https://gecgithub01.walmart.com/Qarth/tensorflow-classifiers',
        'author': 'Alessandro Magnani',
        'author_email': 'AMagnani@walmartlabs.com',
        'license': 'proprietary',
        'classifiers': [ "Private :: Do Not Upload" ],
        'install_requires': [

            # Required for ML and linguistics - confirm with AM (move to sub-package?)
            'numpy',                # required for scikit setup itself (and beyond)
            'scikit-learn',         #==0.14.1
            'scipy',                #==0.12.0,
            'product-status-storage-client',
            #'productdict',
            'protobuf',
            'braavos-data-api==1.0.19'
        ],
        'packages': find_packages(),
    }

    return params

if __name__ == '__main__':
    setup(**configuration())
