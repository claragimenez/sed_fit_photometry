#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools.extension import Extension

import subprocess

import os

#update version
args = 'git describe --tags'
p = subprocess.Popen(args.split(), stdout=subprocess.PIPE)
version = p.communicate()[0].decode("utf-8").strip()

#### Versions
version = "0.1" # init

# Set this to true to add install_requires to setup
if True:
    install_requires=[
         'astropy>=4.0',
         'numpy>=1.17.0',
         'matplotlib>=3.0']
else:
    install_requires = []    
    
#lines = open('grizli/version.py').readlines()
version_str = """# git describe --tags
__version__ = "{0}"\n""".format(version)
fp = open('sed_fit_photometry/version.py','w')
fp.write(version_str)
fp.close()
print('Git version: {0}'.format(version))

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "sed_fit_photometry",
    version = version,
    author = "Clara Giménez Arteaga",
    author_email = "clara.arteaga@nbi.ku.dk",
    description = "HST Paschen Beta analysis",
    license = "MIT",
    url = "https://github.com/claragimenez/sed_fit_photometry/",
    download_url = "https://github.com/claragimenez/sed_fit_photometry/tarball/{0}".format(version),
    packages=['pab'],
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    install_requires=install_requires,
    package_data={} #'sed_fit_photometry': ['data/*header', 'data/psf/*']},
)