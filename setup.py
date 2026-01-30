from setuptools import setup, Extension
import numpy as np


def get_extensions():
    """Get extension modules, returns empty list if Cython not available."""
    try:
        from Cython.Build import cythonize

        extensions = [
            Extension(
                "fpcs.fpcs_cy",
                sources=["src/fpcs/fpcs_cy.pyx"],
                include_dirs=[np.get_include()],
            )
        ]

        return cythonize(
            extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "initializedcheck": False,
                "nonecheck": False,
            },
        )
    except ImportError:
        print("WARNING: Cython not available. Using pure Python implementation.")
        return []


setup(ext_modules=get_extensions())
