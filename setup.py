from setuptools import setup, find_packages
from Cython.Build import cythonize, Extension

# Specify the source directory for Cython files
source_dir = "cython/src"

# Specify the target directory for the build
target_dir = "cython/build"

# Define source files
components = [
    Extension("main", f"{source_dir}/*.pyx"),
    Extension("TradingSim", f"{source_dir}/TradingSim/*.pyx"),
    Extension("Algorithms", f"{source_dir}/Algorithms/*.pyx")
]

setup(
    ext_modules=cythonize(
        components,
        compiler_directives={"language_level": "3"},
    ),
    script_args=["build_ext", "--inplace", f"--build-lib={target_dir}"],
)