import os

# Headless default for servers/CI/report rendering.
# If a user really wants a GUI backend, they can set MPLBACKEND before import.
os.environ.setdefault("MPLBACKEND", "Agg")

__all__ = ["__version__"]
__version__ = "0.1.0"