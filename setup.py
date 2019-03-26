"""
Setup of memory python codebase
Author: Jeff Mahler, Vishal Satish
"""
from setuptools import setup, find_packages
import os

requirements = [
    "autolab-core",
    "keras",
    "tensorflow>=1.10.0",
    "numpy",
    "scikit-image"
]

exec(open("memory/version.py").read())

setup(name="memory", 
      version=__version__, 
      description="Project code for the Dex-Net memory project.", 
      author="Kate Sanders, Vishal Satish", 
      author_email="katesanders@berkeley.edu, vsatish@berkeley.edu", 
      url = "https://github.com/BerkeleyAutomation/memory",
      keywords = "robotics vision deep learning",
      classifiers = [
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 2.7 :: Only',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering'
      ],      
      packages=find_packages(), 
      install_requires = requirements,
)
