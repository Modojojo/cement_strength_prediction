from setuptools import setup

with open('requirements.txt') as f:
    packages = f.read().splitlines()

setup(
    name="concrete-strength-reg",
    version="1.0",
    description="Concrete Strength Prediction package",
    author="Modojojo",
    install_requires=packages
)