from pathlib import Path
import platform

from setuptools import Extension, setup

X86_MACHINES = ["x86_64", "amd64", "x86", "i386", "i686", "AMD64"]
IS_X86 = platform.machine().lower() in X86_MACHINES

UNIX_CXXFLAGS = ["-std=c++17", "-ftree-vectorize"]

# x86-only flags for GCC, not supported by aarch64
UNIX_X86_CXXFLAGS = [
    "-mavx2",
    # GCP N2
    "-march=haswell",
    "-maes",
    "-mno-pku",
    "-mno-sgx",
    "--param", "l1-cache-line-size=64",
    "--param", "l1-cache-size=32",
    "--param", "l2-cache-size=33792",
]


def unix_flags(*extra_x86):
    """Portable flags, plus the Haswell-tuned ones only where they mean something."""
    flags = [*UNIX_CXXFLAGS]
    if IS_X86:
        flags += [*UNIX_X86_CXXFLAGS, *extra_x86]
    return flags


CXX_ARGS = {
    # "Darwin": [*UNIX_CXXFLAGS],  not supported anymore due to M1, PRs welcome
    "Linux": ["-fopenmp", *unix_flags("-mabm")],
    "Windows": ["/openmp", "/std:c++latest", "/arch:AVX2"],
}

project_root = Path(__file__).parent

with open(project_root / "README.md", encoding="utf-8") as f:
    long_description = f.read()


class get_numpy_include:
    """Defer numpy.get_include() until after numpy is installed."""

    def __str__(self):
        import numpy
        return numpy.get_include()


setup(
    name="lapjv",
    description="Linear sum assignment problem solver using Jonker-Volgenant algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.3.27",
    license="MIT",
    author="Vadim Markovtsev",
    author_email="gmarkhor@gmail.com",
    url="https://github.com/src-d/lapjv",
    download_url="https://github.com/src-d/lapjv",
    ext_modules=[Extension("lapjv",
                           sources=["python.cc"],
                           extra_compile_args=CXX_ARGS[platform.system()],
                           include_dirs=[get_numpy_include()])],
    install_requires=["numpy>=1.20.0"],
    tests_require=["scipy>=1.6.0"],
    setup_requires=["numpy>=1.21.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# python3 setup.py bdist_wheel
# auditwheel repair -w dist dist/*
# twine upload dist/*manylinux*
