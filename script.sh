# Compiler Python en C puis binaire
pip install cython

# Créer setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("app1_ofusk.py")
)

# Compiler
python setup.py build_ext --inplace