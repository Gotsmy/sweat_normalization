import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="size_effect_normalization",
    version="1.0",
    author="Mathias Gotsmy",
    author_email="mathias.gotsmy@univie.ac.at",
    description="A python package for size effect normalization in time series metabolome data sets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gotsmy/sweat_normalization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNUv3 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7,<3.8",
    include_package_data=True,
    package_data={'': ['data/*.csv']},
)
