import subprocess
import torch
import os

from setuptools import setup, find_packages
from setuptools.command.install import install
from extensions import get_extensions

class BuildOpsInstallCommand(install):
    """Customized setuptools install command - runs a shell script after install."""

    def run(self):
        install.run(self)
        # Run the shell script
        subprocess.check_call(["./build_ops.sh"])

setup(
    name="CountGD",
    version="0.1.0",
    author="Niki Amini-Naieni, Tengda Han, & Andrew Zisserman",
    description="Forked PyTorch implementation for CountGD.",
    url="https://github.com/landing-ai/CountGD",
    packages=find_packages(),
    install_requires=[
        "cython==3.0.9",
        "submitit==1.5.1",
        "scipy==1.13.1",
        "termcolor==2.4.0",
        "addict==2.4.0",
        "yapf==0.40.1",
        "timm==0.9.16",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.42.3",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "pycocotools==2.0.7",
        "pyyaml==6.0.1",
        "colorlog==6.8.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    # cmdclass={"install": BuildOpsInstallCommand},
)
