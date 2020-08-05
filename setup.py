from setuptools import setup, find_packages
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_version():
    with open("fcmaesray/__init__.py", "r") as f:
        match = re.search(r"(?m)^__version__\s*=\s*['\"](.+)['\"]$", f.read())
        return match.group(1)

setup(
    name='fcmaesray',
    version=get_version(),
    description=('A multi-node Python 3 gradient-free optimization library.'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Dietmar Wolz',
    author_email='drdietmarwolz@yahoo.de',
    url='https://github.com/dietmarwo/fcmaes-ray',
    license='MIT',
    packages=find_packages(),
    install_requires=[
          'ray', 'fcmaes'
    ],
    classifiers = [
      "Intended Audience :: Science/Research",
      "Intended Audience :: Education",
      "Intended Audience :: Manufacturing",
      "Intended Audience :: Other Audience",
      "Topic :: Scientific/Engineering",
      "Topic :: Scientific/Engineering :: Mathematics",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
      "Operating System :: OS Independent",
      "Programming Language :: Python :: 3",
      "Development Status :: 4 - Beta",
      "Environment :: Console",
      "License :: OSI Approved :: MIT License",
      ],
    keywords=["optimization", "CMA-ES", "differential evolution", "stochastic", "gradient free", "parallel retry", "multi node", "ray", "fcmaes"],
    include_package_data=True,
   )
