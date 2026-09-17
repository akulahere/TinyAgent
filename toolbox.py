"""Small, explicit Python functions that can be registered as agent tools."""


def add(a: str, b: str) -> float:
    """Add two numbers provided as strings."""
    return float(a) + float(b)


def subtract(a: str, b: str) -> float:
    """Subtract b from a, with both numbers provided as strings."""
    return float(a) - float(b)


def multiply(a: str, b: str) -> float:
    """Multiply two numbers provided as strings."""
    return float(a) * float(b)
