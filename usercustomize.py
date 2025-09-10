"""
Ensure pytest third-party plugin autoload is disabled, even when a system
sitecustomize shadows the project-local one. Python imports usercustomize after
sitecustomize if present on sys.path.
"""
import os as _os

_os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

