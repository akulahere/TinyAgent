"""Small, explicit Python functions that can be registered as agent tools."""

from datetime import date


def add(a: str, b: str) -> float:
    """Add two numbers provided as strings."""
    return float(a) + float(b)


def subtract(a: str, b: str) -> float:
    """Subtract b from a, with both numbers provided as strings."""
    return float(a) - float(b)


def multiply(a: str, b: str) -> float:
    """Multiply two numbers provided as strings."""
    return float(a) * float(b)


def today() -> str:
    """Return today's date in the host's local timezone (YYYY-MM-DD)."""
    return date.today().isoformat()


def days_between(a: str, b: str) -> int:
    """Return b minus a in days for ISO dates; equal dates give zero."""
    return (date.fromisoformat(b) - date.fromisoformat(a)).days
