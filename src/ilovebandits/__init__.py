"""Configure submodules say the library what to do when calling "ilovebandits.submodule."""
import importlib as _importlib

# Configure submodules say the library what to do when calling "ilovebandits.submodule"
# base on __init__ file of sklearn: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/__init__.py#L19
_submodules = [
    "agents",
    "sim",
    "data_bandits",
]

def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"ilovebandits.{name}")
    else:
        raise AttributeError(f"Module 'ilovebandits' has no attribute '{name}'")
