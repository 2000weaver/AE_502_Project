import importlib
import pkgutil
from pathlib import Path


def _import_package_symbols(package_name: str, package_path: Path) -> list[str]:
    exports = []
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        names = getattr(module, "__all__", [n for n in dir(module) if not n.startswith("_")])
        for name in names:
            if name in globals():
                continue
            globals()[name] = getattr(module, name)
            exports.append(name)
    return exports


__all__ = _import_package_symbols(__name__, Path(__file__).parent)
