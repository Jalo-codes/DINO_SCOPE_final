"""lab_utils.errors — typed exceptions used throughout the library.

Raise these instead of bare ValueError/RuntimeError so callers can
catch specific failure modes without accidentally swallowing unrelated
exceptions.
"""


class DataError(Exception):
    """Raised when input data violates an expected shape, type, or range."""


class ConfigError(Exception):
    """Raised when a configuration is self-inconsistent or incompatible with
    the runtime environment (e.g., AE cache built at wrong resolution)."""


class EvalError(Exception):
    """Raised when an evaluation run cannot proceed (missing predictions,
    incompatible metric, ill-formed stratification keys)."""
