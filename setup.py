from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="circular-grid-detector",
    version="1.0.0",
    author="Shubham Agarwal",
    author_email="sragarwal@outlook.in",
    description="High-performance circular grid pattern detection library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crazysa/circular-grid-detector",
    project_urls={
        "Bug Tracker": "https://github.com/crazysa/circular-grid-detector/issues",
        "Documentation": "https://github.com/crazysa/circular-grid-detector/wiki",
        "Source Code": "https://github.com/crazysa/circular-grid-detector",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    keywords=[
        "computer-vision",
        "circle-detection", 
        "grid-pattern",
        "calibration",
        "opencv",
        "image-processing",
        "pattern-recognition"
    ],
    include_package_data=True,
    zip_safe=False,
)
