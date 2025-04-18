from setuptools import setup, find_packages

setup(
    name="minigpt",
    version="0.1.0",
    description="A minimal implementation of GPT, inspired by Andrej Karpathy's minGPT",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/minigpt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "regex>=2021.8.3",
        "requests>=2.26.0",
        "pyyaml>=5.4.1",
    ],
    entry_points={
        "console_scripts": [
            "minigpt=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
