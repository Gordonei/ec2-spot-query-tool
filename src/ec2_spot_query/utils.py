"""Generic helper utilities shared across the package.

These are pure, dependency-free functions with no EC2/``boto3`` concerns,
kept separate from :mod:`ec2_spot_query.core` (which holds the business
logic).
"""

from __future__ import annotations

import fnmatch
from typing import Any

# Official AWS EC2 processor architecture values
OFFICIAL_ARCHITECTURES = frozenset({
    "x86_64",
    "arm64",
    "x86_64_mac",
    "arm64_mac",
    "riscv64",
})

# User-friendly aliases -> official architecture values
ARCH_ALIASES: dict[str, tuple[str, ...]] = {
    "x86": ("x86_64",),
    "intel": ("x86_64",),
    "amd": ("x86_64",),
    "arm": ("arm64",),
    "graviton": ("arm64",),
    "aarch64": ("arm64",),
    "mac": ("arm64_mac", "x86_64_mac"),
}


def map_architectures(arch_input: str | list[str] | None) -> list[str] | None:
    """Map user-friendly architecture aliases to official AWS EC2 values.

    Accepts a single string or a list of strings (case-insensitive).
    ``None`` returns ``None`` (no architecture filter).

    Aliases:
        x86, x86_64, intel, amd  -> x86_64
        arm, arm64, graviton, aarch64 -> arm64
        mac                      -> arm64_mac, x86_64_mac

    Official AWS values (x86_64, arm64, x86_64_mac, arm64_mac, riscv64)
    pass through unchanged.  Results are deduplicated, preserving order.

    Raises:
        ValueError: If a token is neither a known alias nor an official
            architecture value.
    """
    if arch_input is None:
        return None
    if isinstance(arch_input, str):
        tokens = [arch_input]
    else:
        tokens = list(arch_input)

    mapped: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        if not key:
            continue
        if key in OFFICIAL_ARCHITECTURES:
            values = (key,)
        elif key in ARCH_ALIASES:
            values = ARCH_ALIASES[key]
        else:
            valid = ", ".join(sorted(set(ARCH_ALIASES) | OFFICIAL_ARCHITECTURES))
            raise ValueError(f"Unknown architecture {token!r}. Valid values: {valid}")
        for value in values:
            if value not in mapped:
                mapped.append(value)
    return mapped or None


_GLOB_CHARS = frozenset("*?[")


def has_glob(pattern: str) -> bool:
    """Return True if *pattern* contains fnmatch glob characters (*, ?, [)."""
    return any(char in _GLOB_CHARS for char in pattern)


def match_names(names: list[str], pattern: str) -> list[str]:
    """Return the names matching *pattern* (case-sensitive fnmatch), sorted."""
    return sorted(name for name in names if fnmatch.fnmatchcase(name, pattern))


def expand_names(names: list[str], patterns: list[str]) -> list[str]:
    """Expand *patterns* against *names*.

    Patterns containing glob characters are matched case-sensitively against
    *names*; patterns without glob characters pass through as-is (even if
    absent from *names*).  Returns the deduplicated, sorted union.
    """
    matched: set[str] = set()
    for pattern in patterns:
        if not pattern:
            continue
        if has_glob(pattern):
            matched.update(match_names(names, pattern))
        else:
            matched.add(pattern)
    return sorted(matched)


def get_retry_count(resp_meta: dict | None) -> int:
    """Extract retry count from response metadata, safe for MagicMock in tests."""
    if resp_meta is None:
        return 0
    val = resp_meta.get("RetryAttempts")
    return val if isinstance(val, int) else 0


def validate_range(constraint_name: str, min_val: Any, max_val: Any) -> None:
    """Raise ``ValueError`` if *max_val* is less than *min_val*.

    *constraint_name* is used to build a readable error message, e.g.
    ``validate_range("vcpu", 8, 4)`` raises
    ``ValueError("max_vcpu (4) cannot be less than min_vcpu (8)")``.
    """
    if max_val < min_val:
        raise ValueError(
            f"max_{constraint_name} ({max_val}) cannot be less than "
            f"min_{constraint_name} ({min_val})"
        )
