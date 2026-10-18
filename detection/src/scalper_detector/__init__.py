"""Ticket-only scalper scoring from cross-site payment activity."""

from .runtime import configure_environment

# A console entry point imports this package before its CLI and native dependencies.
configure_environment()

__version__ = "0.3.0"
