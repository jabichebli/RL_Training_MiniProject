try:
    from mjlab.third_party.isaaclab.isaaclab_tasks.utils.importer import import_packages

    _BLACKLIST_PKGS = ["utils", ".mdp"]

    import_packages(__name__, _BLACKLIST_PKGS)
except (ImportError, ModuleNotFoundError, AttributeError):
    # Fallback: manually import task configs if automatic import fails
    # This can happen if mjlab isn't fully installed or in certain import contexts
    try:
        # Use relative imports since we're inside the tasks package
        from .velocity.config import go1, g1
        from .tracking.config import g1 as tracking_g1
    except (ImportError, AttributeError):
        # If relative imports fail, try absolute imports as last resort
        try:
            import mjlab.tasks.velocity.config.go1
            import mjlab.tasks.velocity.config.g1
            import mjlab.tasks.tracking.config.g1
        except ImportError:
            pass  # Task configs will be imported when needed
