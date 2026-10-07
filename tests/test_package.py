"""Tests for ec2_spot_query package metadata."""

from __future__ import annotations

from ec2_spot_query import __version__


def test_version_exists():
    """__version__ is defined."""
    assert isinstance(__version__, str)


def test_docstring_uses_british_spelling():
    """Module docstring uses 'analyse' (British English) not 'analyze'."""
    import ec2_spot_query
    doc = ec2_spot_query.__doc__ or ""
    assert "analyse" in doc, "Module docstring should use British spelling 'analyse'"
    assert "analyze" not in doc, "Module docstring should not use American spelling 'analyze'"
