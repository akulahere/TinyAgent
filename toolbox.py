"""Small, explicit Python functions that can be registered as agent tools."""


def multiply(a: str, b: str) -> float:
    """Multiply two numbers provided as strings."""
    return float(a) * float(b)
