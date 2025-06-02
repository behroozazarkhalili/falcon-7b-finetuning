"""Setup script for Falcon-7B Fine-tuning Project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="falcon-7b-finetuning",
    version="1.0.0",
    author="Behrooz Azarkhalili",
    author_email="ermiaazarkhalili@gmail.com",
    description="A production-ready ML engineering project for fine-tuning Falcon-7B using QLoRA and TRL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/behroozazarkhalili/falcon-7b-finetuning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "falcon-train=scripts.train:main",
            "falcon-evaluate=scripts.evaluate:main",
            "falcon-inference=scripts.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 