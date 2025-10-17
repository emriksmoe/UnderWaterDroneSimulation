from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="underwater-dtn-simulation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Underwater DTN simulation for Age of Information research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/UnderWaterDroneSimulation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "analysis": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "underwater-sim=src.simulation.runner:main",
        ],
    },
)