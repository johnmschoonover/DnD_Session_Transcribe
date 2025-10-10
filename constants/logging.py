"""Logging configuration constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoggingConfig:
    """Default logging configuration for the CLI."""

    level: str = "WARNING"


__all__ = ["LoggingConfig"]
