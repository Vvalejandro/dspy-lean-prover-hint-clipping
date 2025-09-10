"""
Ensure vanilla `pytest` runs cleanly by disabling auto-loaded third-party
plugins that may be present in the execution environment.

Placing this in the repo root leverages Python's sitecustomize import hook to
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for any Python process started from here.
"""
import os as _os

_os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

