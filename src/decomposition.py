"""CEEMDAN decomposition and variable-length alignment helpers."""

try:
    from .rolling_ceemdan_core import (
        decompose_ceemdan,
        rolling_ceemdan,
        select_first_k,
    )
except ImportError:  # Allows direct execution from notebooks.
    from rolling_ceemdan_core import (
        decompose_ceemdan,
        rolling_ceemdan,
        select_first_k,
    )

__all__ = ["decompose_ceemdan", "rolling_ceemdan", "select_first_k"]
