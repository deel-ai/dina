from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Dina",
    version="0.0.1",
    description="DEEL ImageNet Attributions datasets and benchmarking",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Lucas Hervier",
    author_email="lucas.hervier@irt-saintexupery.com",
    license="MIT",
    install_requires=["tensorflow==2.15.0", "kecam==1.4.1", "xplique==1.3.3", "tensorflow_probability==0.23.0",
                      "pyyaml==6.0.2", "einops==0.8.1", "tqdm==4.66.5"],
    packages=find_packages(),
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
