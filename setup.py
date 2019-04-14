"""
Setup of memory python codebase.

Author: Vishal Satish, Jeff Mahler
"""
import os
import sys
import logging
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

TF_MIN_VERSION = "1.10.0"
TF_MAX_VERSION = "1.13.1"

# set up logger
logging.basicConfig() # configure the root logger
logger = logging.getLogger("setup.py")
logger.setLevel(logging.INFO)

def get_tf_dep():
    # check whether or not the Nvidia driver and GPUs are available and add the corresponding Tensorflow dependency
    tf_dep = "tensorflow>={},<={}".format(TF_MIN_VERSION, TF_MAX_VERSION)
    try:
        gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"]).decode().strip().split('\n')[1:]
        if len(gpus) > 0:
            tf_dep = "tensorflow-gpu>={},<={}".format(TF_MIN_VERSION, TF_MAX_VERSION)
        else:
            logger.warning("Found Nvidia device driver but no devices...installing Tensorflow for CPU.")
    except OSError:
        logger.warning("Could not find Nvidia device driver...installing Tensorflow for CPU.")
    return tf_dep

#TODO(vsatish): Use inheritance here
class DevelopCmd(develop):
    user_options_custom = [
        ("docker", None, "installing in Docker"),
    ]
    user_options = getattr(develop, "user_options", []) + user_options_custom

    def initialize_options(self):
        develop.initialize_options(self)

        # initialize options
        self.docker = False

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        # install Tensorflow dependency
        if not self.docker:
            tf_dep = get_tf_dep()
            subprocess.Popen([sys.executable, "-m", "pip", "install", tf_dep]).wait()
        else:
            # if we're using docker, this will already have been installed explicitly through the correct {cpu/gpu}_requirements.txt; there is no way to check for CUDA/GPUs at docker build time because there is no easy way to set the nvidia runtime
            logger.warning("Omitting Tensorflow dependency because of Docker installation.") #TODO(vsatish): Figure out why this isn't printed

        # run installation
        develop.run(self)

class InstallCmd(install, object):
    user_options_custom = [
        ("docker", None, "installing in Docker"),
    ]
    user_options = getattr(install, "user_options", []) + user_options_custom

    def initialize_options(self):
        install.initialize_options(self)

        # initialize options
        self.docker = False

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # install Tensorflow dependency
        if not self.docker:
            tf_dep = get_tf_dep()
            subprocess.Popen([sys.executable, "-m", "pip", "install", tf_dep]).wait()
        else:
            # if we're using docker, this will already have been installed explicitly through the correct {cpu/gpu}_requirements.txt; there is no way to check for CUDA/GPUs at docker build time because there is no easy way to set the nvidia runtime
            logger.warning("Omitting Tensorflow dependency because of Docker installation.") #TODO(vsatish): Figure out why this isn't printed

        # run installation
        install.run(self)

requirements = [
    "autolab-core",
    "numpy>=1.14.0",
    "scikit-image<0.15.0",
    "keras",
    "nearpy"
]

exec(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "memory/version.py")).read())

setup(name="memory", 
      version=__version__, 
      description="Project code for the Dex-Net memory project.", 
      author="Kate Sanders, Vishal Satish", 
      author_email="katesanders@berkeley.edu, vsatish@berkeley.edu", 
      url = "https://github.com/BerkeleyAutomation/memory",
      keywords = "robotics vision deep learning",
      classifiers = [
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 2.7 :: Only",
          "Natural Language :: English",
          "Topic :: Scientific/Engineering"
      ],      
      packages=find_packages(), 
      install_requires = requirements,
      cmdclass={
        "install": InstallCmd,
        "develop": DevelopCmd
      }
)

