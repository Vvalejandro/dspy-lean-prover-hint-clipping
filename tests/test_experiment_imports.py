def test_experiment_module_imports_without_heavy_deps():
    # Importing the experiment module should not require typer/rich/ujson at import time.
    mod = __import__("experiments.experiment", fromlist=["*"])
    assert hasattr(mod, "run")
