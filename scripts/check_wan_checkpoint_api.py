"""CPU-only compatibility check against the installed Diffusers API; no weights loaded."""
import os
import ast
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diffusers
from diffusers.models.modeling_utils import ModelMixin

root = Path(__file__).resolve().parents[1]
# Importing the upstream wan package initializes CUDA in T5's default argument.
# Bind the exact source hook to the actual installed ModelMixin on CPU instead.
for name, filename in (("WanModel", "model.py"), ("CausalWanModel", "causal_model.py")):
    source = root / "wan/modules" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    hook = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_set_gradient_checkpointing")
    namespace = {}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec"), namespace)
    model_type = type(name, (ModelMixin,), {"_supports_gradient_checkpointing": True,
                       "_set_gradient_checkpointing": namespace["_set_gradient_checkpointing"]})
    model = model_type()
    model.gradient_checkpointing = False
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing is True
    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False
    print(f"{name}: Diffusers {diffusers.__version__} enable/disable passed")
