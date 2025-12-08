"""Tests for IMFProtocol type checking."""
import pytest
from progenax.imf.base import IMFProtocol
from progenax.imf import ChabrierIMF, PowerLawIMF, TruncatedIMF, Maschberger


class TestIMFProtocol:
    """Verify all IMFs satisfy IMFProtocol."""

    def test_chabrier_is_imf_protocol(self):
        """ChabrierIMF satisfies IMFProtocol."""
        imf = ChabrierIMF()
        assert isinstance(imf, IMFProtocol)

    def test_powerlaw_is_imf_protocol(self):
        """PowerLawIMF satisfies IMFProtocol."""
        imf = PowerLawIMF.salpeter()
        assert isinstance(imf, IMFProtocol)

    def test_truncated_is_imf_protocol(self):
        """TruncatedIMF satisfies IMFProtocol."""
        inner = ChabrierIMF()
        imf = TruncatedIMF(inner, m_min=0.1, m_max=50.0)
        assert isinstance(imf, IMFProtocol)

    def test_maschberger_is_imf_protocol(self):
        """Maschberger satisfies IMFProtocol."""
        imf = Maschberger()
        assert isinstance(imf, IMFProtocol)
