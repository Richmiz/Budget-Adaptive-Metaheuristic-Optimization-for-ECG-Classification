"""Verify notebook code-cell syntax, embedded ZIP hash and current source bytes."""
import _paths  # noqa: F401

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from bamo.config import PROJECT_ROOT
from bamo.utils import sha256_file, write_json
from build_colab_notebook import build_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/COLAB_G4_DECISION.ipynb")
    ap.add_argument("--out", default="docs/evidence/notebook_bundle_verification.json")
    args = ap.parse_args()
    path = PROJECT_ROOT / args.notebook
    nb = json.loads(path.read_text(encoding="utf-8"))
    constants = {}
    n_code = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("BUNDLE_B64", "BUNDLE_SHA"):
                        constants[target.id] = node.value.value
    bundle = base64.b64decode(constants["BUNDLE_B64"], validate=True)
    digest = hashlib.sha256(bundle).hexdigest()
    if digest != constants["BUNDLE_SHA"]:
        raise ValueError("notebook ZIP hash mismatch")
    current, names = build_bundle()
    if bundle != current:
        raise ValueError("embedded code differs from current source; rebuild the notebook")
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        if archive.namelist() != names:
            raise ValueError("bundle inventory differs")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("unsafe bundle path")
            if archive.read(name) != (PROJECT_ROOT / name).read_bytes():
                raise ValueError(f"source bytes differ: {name}")
    result = {"notebook": args.notebook, "notebook_sha256": sha256_file(path),
              "bundle_sha256": digest, "n_source_files": len(names), "n_code_cells_parsed": n_code,
              "source_bytes_match": True, "limits": "Static packaging verification; G4 execution pending"}
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
