"""Run build_season_patches.py without PyTorch / terratorch installed.

build_season_patches.py uses torch only to read each labelled site's id from data/processed/*.pt, and imports three
small helper functions from pipeline modules that pull torch in at the top. If torch is importable this just runs
the script unchanged. Otherwise it (1) reads the ids straight from the .pt files (zip + pickle) and (2) lifts the
three helpers out of the pipeline source files, unchanged, so nothing else has to be imported.
Usage: python3 bg_build_notorch.py <same arguments as build_season_patches.py>
"""
import ast, math, os, pickle, runpy, sys, types, zipfile
from pathlib import Path

sys.path.insert(0, os.getcwd())

try:
    import torch  # noqa: F401
except Exception:
    import json
    import numpy as np
    import rasterio
    from rasterio.warp import transform as warp_transform
    from utils.config import load_config

    def _read_pt(path):
        class U(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod.startswith("torch"):
                    return lambda *a, **k: None
                return super().find_class(mod, name)
            def persistent_load(self, pid):
                return None
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
            return U(z.open(name)).load()

    cfg = load_config()

    def lift(path, func):
        src = open(path).read()
        tree = ast.parse(src)
        ns = {"math": math, "np": np, "json": json, "Path": Path, "rasterio": rasterio,
              "warp_transform": warp_transform, "cfg": cfg}
        for node in tree.body:   # module-level constants such as REGION_KM
            if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                try:
                    exec(compile(ast.Module([node], []), path, "exec"), ns)
                except Exception:
                    pass
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func:
                exec(compile(ast.Module([node], []), path, "exec"), ns)
                return ns[func]
        raise SystemExit(f"{func} not found in {path}")

    for modname, path, func in [("pipeline.crossval", "pipeline/crossval.py", "overlap_groups"),
                                ("pipeline.evaluate", "pipeline/evaluate.py", "load_site_coords"),
                                ("pipeline.preprocess", "pipeline/preprocess.py", "find_tile_containing_point")]:
        m = types.ModuleType(modname)
        setattr(m, func, lift(path, func))
        sys.modules[modname] = m
    t = types.ModuleType("torch")
    t.load = _read_pt
    sys.modules["torch"] = t
    print("(torch not installed: ids read directly from the .pt files; three helper functions lifted from pipeline/)")

script = os.path.join(os.getcwd(), "build_season_patches.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
