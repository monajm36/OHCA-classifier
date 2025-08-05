from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ohca-classifier",
    version="1.0.0",
    author="monajm36",
    author_email="mjm36@uchicago.edu",
    description="A BERT-based classifier for Out-of-Hospital Cardiac Arrest (OHCA) detection in medical text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monajm36/OHCA-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="medical-ai, nlp, bert, classification, healthcare, cardiac-arrest",
    project_urls={
        "Bug Reports": "https://github.com/monajm36/OHCA-classifier/issues",
        "Source": "https://github.com/monajm36/OHCA-classifier",
        "Documentation": "https://github.com/monajm36/OHCA-classifier#readme",
    },
)
