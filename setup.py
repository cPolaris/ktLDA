import setuptools
from distutils.extension import Extension
# from distutils.core import setup
# from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ktLDA",
    version="0.0.6",
    author="Kehan (kehanLyu) & Tiangang (cPolaris)",
    author_email="",
    description="An implementation of latent Dirichlet allocation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cPolaris/ktLDA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=[Extension("cgibbs_inf", ["ktlda/cgibbs_inf.c"])]
)

# setup(
#     ext_modules=cythonize("ktlda/cgibbs_inf.pyx")
# )
