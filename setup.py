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

# print(find_packages(include=["countgd"]))
# breakpoint()
setup(
    name="countgd",
    version="0.1.0",
    author="Niki Amini-Naieni, Tengda Han, & Andrew Zisserman",
    description="Forked PyTorch implementation for CountGD.",
    url="https://github.com/landing-ai/CountGD",
    # packages=[
    #     "countgd.datasets_inference",
    #     "countgd.models",
    #     "countgd.models_inference",
    #     "countgd.models_inference.GroundingDINO"
    #     "countgd.models_inference.GroundingDINO.backbone"
    #     "countgd.util",
    # ],
    # packages=find_packages(include=["countgd", "countgd.*"]),
    packages=find_packages(include=["countgd"], exclude=["countgd.*.*"]),
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
    python_requires=">=3.10",
    # package_dir={
    #     "countgd.datasets_inference": "datasets_inference",
    #     "countgd.models": "models",
    #     "countgd.models_inference": "models_inference",
    #     "countgd.util": "util",
    #     "countgd.groundingdino": "groundingdino",
    # },
    package_dir={"countgd": "countgd"},
    # cmdclass={"install": BuildOpsInstallCommand},
)
