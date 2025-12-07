"""Initial Mass Functions (IMFs) for stellar population synthesis."""

from .base import BaseIMF, _ppf_newton
from .truncated import TruncatedIMF

__all__ = ["BaseIMF", "_ppf_newton", "TruncatedIMF"]
