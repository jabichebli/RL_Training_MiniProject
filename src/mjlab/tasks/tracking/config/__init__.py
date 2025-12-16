# Import config modules to register tasks
# Use importlib to avoid circular import issues during package initialization
def _import_configs():
  """Lazy import of config modules to register tasks."""
  import importlib
  importlib.import_module("mjlab.tasks.tracking.config.g1")
  importlib.import_module("mjlab.tasks.tracking.config.go2")

# Import immediately to register tasks
_import_configs()

