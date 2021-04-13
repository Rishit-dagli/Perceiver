from setuptools import setup

exec(open("perceiver/version.py").read())

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="perceiver",
    version="0.1.0",
    description="Implement of Perceiver, General Perception with Iterative Attention in TensorFlow",
    packages=["perceiver"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "perceiver",
        "artificial intelligence",
        "deep learning",
        "transformer",
        "attention mechanism",
    ],
    url="https://github.com/Rishit-dagli/Perceiver",
    author="Rishit Dagli",
    author_email="rishit.dagli@gmail.com",
    install_requires=[
        "tensorflow~=2.4.0",
        "einops>=0.3",
    ],
    extras_require={
        "dev": ["check-manifest", "twine", "numpy"],
    },
)
