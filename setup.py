import runpy

from setuptools import find_packages, setup

version = runpy.run_path("limbresml/__init__.py")["__version__"]

setup(
    name="limbresml",
    version=version,
    author="Tai-Yu Pan and Mengdi Fan",
    url="https://github.com/Rescue-Heroes/LimbRescueML",
    description="A machine learning library for Project Limb Rescue",
    python_requires=">=3.8",
    install_requires=["scikit-learn", "tqdm", "matplotlib", "pandas", "iopath", "yacs"],
    packages=find_packages(exclude=("configs", "data", "rawdata")),
)
