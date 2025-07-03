from setuptools import setup, find_packages

setup(
    name="nami_act",
    version="1.0.0",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="Official Repository for Nami: an adaptive self regulatory activation function",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/Nami.git",
    packages=find_packages(
        where=".",
        exclude=["nami.Torch.functional"]
    ),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=False,
    
)