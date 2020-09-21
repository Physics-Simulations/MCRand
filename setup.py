import setuptools
from mcrand import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcrand",
    version=__version__,
    author="Physics Simulations",
    author_email="apuntsdefisica@gmail.com",
    description="MCRand is a library of Monte Carlo methods implementing multidimensional integration \
                 and non-uniform random number generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Physics-Simulations/MCRand",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.5'
    ],
    python_requires='>=3.6',
)
