import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stannetflow",
    version="0.0.1",
    author="Shengzhe Xu",
    author_email="shengzx@vt.edu",
    description="top-level package for netflow-stan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShengzheXu/stan",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy==1.19.2",
        "pandas==1.2.0",
        "scikit-learn==0.24.0",
        "matplotlib==3.3.3",
        "pytorch==1.7.1",
    ],
)