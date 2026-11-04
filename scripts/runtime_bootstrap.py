"""Load OpenPI's required Transformers overrides without editing the shared venv."""
import importlib.abc
import importlib.metadata
import importlib.util
from pathlib import Path
import sys


def install_transformers_overrides(openpi_root):
    if importlib.metadata.version("transformers") != "4.53.2":
        raise RuntimeError("This OpenPI checkout requires transformers==4.53.2")
    root = Path(openpi_root) / "src/openpi/models_pytorch/transformers_replace"
    modules = {"transformers." + ".".join(path.relative_to(root).with_suffix("").parts): path
               for path in root.rglob("*.py")}

    class OpenPIOverrides(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in modules:
                return importlib.util.spec_from_file_location(fullname, modules[fullname])
            return None

    sys.meta_path.insert(0, OpenPIOverrides())
