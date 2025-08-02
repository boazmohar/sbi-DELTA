from setuptools import setup, find_packages

setup(
    name="sbi_delta",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sbi",
    ]
)