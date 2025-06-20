from setuptools import setup, find_packages

setup(
    name="Nami",
    version="0.0.1",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="Official Repository for Nami: an adaptive self regulatory activation function",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/cpy-deriv.git",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    zip_safe=False,
    
)