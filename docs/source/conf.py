# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

#### path for Autodoc configuration
## in azure devops I needed:
# sys.path.insert(0, os.path.abspath('../../src/ilovebandits')) # It is important to note here that the absolute path must be specified in relation to where conf.py resides, i.e. our `Sphinx source root`
## in wsl project I needed:
sys.path.insert(0, os.path.abspath("../../src"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ilovebandits"
copyright = "2025, Abel Sancarlos"
author = "Abel Sancarlos"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # 'sphinx.ext.napoleon',  # Optional, for Google-style docstrings
    # 'sphinx.ext.viewcode',  # Optional, adds [source] links
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # default one: 'alabaster'
html_static_path = ["_static"]
