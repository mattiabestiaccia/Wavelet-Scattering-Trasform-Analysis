from setuptools import setup, find_packages

setup(
    name="wavelet_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "matplotlib",
        "tqdm"
    ]
)