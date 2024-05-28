import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

# Read the contents of requirements.txt file
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setuptools.setup(
    name="hwgptbench",
    version="0.1",
    author="AutoML Freiburg",
    author_email="sukthank@cs.uni-freiburg.de",
    description=(
        "A surrogate benchmark for multi-objective neural architecture search."
    ),
    long_description=long_description,
    url="https://github.com/automl/HW-Aware-LLM-Bench",
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    keywords="machine learning"
    "surrogate benchmark gpt hardware-aware NAS deep learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.6",
    platforms=["Linux"],
    install_requires=required_packages,
    include_package_data=True,
    # extras_require=optional_requirements
)
