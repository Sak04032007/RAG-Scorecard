from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="rag-scorecard",
    version="1.0.0",
    author="Kenobi",
    description="A lightweight, resilient, and production-ready evaluation engine to audit Retrieval-Augmented Generation (RAG) pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "openai",
        "google-genai"
    ],
)
