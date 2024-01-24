from setuptools import setup
from Cython.Build import cythonize

# Specify the target directory for the build
target_dir = "cython/build"

setup(
    ext_modules=cythonize(
        "DataFeed.pyx",
        compiler_directives={"language_level": "3"},
    ),
    script_args=["build_ext", "--inplace", f"--build-lib={target_dir}"],
)