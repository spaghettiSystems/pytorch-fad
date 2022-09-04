from setuptools import find_packages, setup, find_namespace_packages

setup(
    name="pytorch_fad",
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src', include='environment_manager.*'),
    version="0.0.2",
    license="MIT",
    description="Fréchet Audio Distance",
    long_description_content_type="text/markdown",
    author="Mohamed Osman",
    author_email="mohamed.osman@ieee.org",
    url="https://github.com/spaghettiSystems/pytorch-fad",
    keywords=["artificial intelligence", "deep learning", "audio generation"],
    install_requires=[
        "torch>=1.6",
        "scipy",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)