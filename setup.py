from pathlib import Path
import platform

from setuptools import Extension, setup

CXX_ARGS = {
    "Darwin": ["-std=c++11", "-march=native", "-ftree-vectorize"],
    "Linux": ["-fopenmp", "-std=c++11", "-march=native", "-ftree-vectorize"],
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
    version="1.3.10",
    license="MIT",
    author="Vadim Markovtsev",
    author_email="gmarkhor@gmail.com",
    url="https://github.com/src-d/lapjv",
    download_url="https://github.com/src-d/lapjv",
    ext_modules=[Extension("lapjv",
                           sources=["python.cc"],
                           extra_compile_args=CXX_ARGS[platform.system()],
                           include_dirs=[get_numpy_include()])],
    install_requires=["numpy>=1.0.0"],
    tests_require=["scipy>=1.0.0"],
    setup_requires=["numpy>=1.0.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python3 setup.py bdist_wheel
# auditwheel repair -w dist dist/*
# twine upload dist/*manylinux*
