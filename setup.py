from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]


EXTRA_REQUIREMENTS = {
    "dali": ["nvidia-dali-cuda110"],
    "umap": ["matplotlib", "seaborn", "pandas", "umap-learn"],
}


def parse_requirements(path):
    with open(path) as f:
        requirements = [p.strip().split()[-1] for p in f.readlines()]
    return requirements


setup(
    name="adios",
    packages=find_packages(exclude=["bash_files"]),
    version="0.0.1",
    license="MIT",
    author="Yuge Shi, Adam Kosiorek",
    author_email="yshi@robots.ox.ac.uk, adamrk@deepmind.com",
    url="https://github.com/yugeten/adios",
    keywords=KW,
    install_requires=parse_requirements("requirements.txt"),
    extras_require=EXTRA_REQUIREMENTS,
    dependency_links=["https://developer.download.nvidia.com/compute/redist"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
