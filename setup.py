import os

from setuptools import find_packages, setup


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setup(
    name="spateo-release",
    version="1.1.2",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("dev-requirements.txt"),
        "docs": read_requirements(os.path.join("docs", "requirements.txt")),
        "3d": read_requirements("3d-requirements.txt"),
    },
    packages=find_packages(exclude=("tests", "tests.*", "docs", "docs.*")),
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    author="Xiaojie Qiu",
    author_email="xqiu@wi.mit.edu",
    description="Spatiotemporal modeling of molecular holograms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    url="https://github.com/gmhhhhhh-929/spateo-release",
    keywords=[
        "spatial-transcriptomics",
        "stereo-seq",
        "Visium",
        "seqFish",
        "MERFISH",
        "slide-seq",
        "DBiT-seq",
        "HDST-seq",
        "osmFISH",
        "spatiotemporal",
    ],
)
