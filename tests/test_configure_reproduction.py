"""CPU-only checks for the portable config generator; no real assets required."""

import importlib.util
from pathlib import Path, PurePosixPath
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "public_repro/scripts/configure_reproduction.py"
if not SCRIPT.is_file():
    SCRIPT = ROOT / "scripts/configure_reproduction.py"
SPEC = importlib.util.spec_from_file_location("configure_reproduction", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class ConfigureReproductionTests(unittest.TestCase):
    def test_root_discovery(self):
        self.assertEqual(generator.repository_root(), ROOT)

    def test_rebind_preserves_recipe_and_correct_longforcing_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "explicit-data-root"
            output = Path(temporary) / "generated"
            paths = generator.configure(root, output)
            configs = {stage: yaml.safe_load(path.read_text(encoding="utf-8")) for stage, path in paths.items()}
            self.assertEqual(set(paths), {"action", "causal", "longforcing", "eval"})
            self.assertFalse(root.exists(), "configuration must not create data or model artifacts")
            for stage, steps in (("action", 1075), ("causal", 1075), ("longforcing", 80)):
                config = configs[stage]
                original = yaml.safe_load((ROOT / generator.TEMPLATES[stage]).read_text(encoding="utf-8"))
                self.assertEqual(config["training"]["max_steps"], steps)
                self.assertEqual(config["model"]["base_model_path"], str(root / "models" / generator.PINNED_MODEL_DIR))
                self.assertEqual(config["data"]["manifest_path"], str(root / "data/manifests/train.jsonl"))
                self.assertIsNone(config["data"]["manifest_sha256"])
                self.assertEqual(config["optimizer"], original["optimizer"])
                for section, excluded in (("model", {"base_model_path"}), ("training", {"output_dir"})):
                    self.assertEqual(
                        {key: value for key, value in config[section].items() if key not in excluded},
                        {key: value for key, value in original[section].items() if key not in excluded},
                    )
            runs = {stage: Path(configs[stage]["training"]["output_dir"]) for stage in ("action", "causal", "longforcing")}
            self.assertEqual(configs["causal"]["lineage"]["checkpoint_path"], str(runs["action"] / "checkpoints/best.pt"))
            self.assertEqual(configs["longforcing"]["lineage"]["teacher_checkpoint_path"], str(runs["action"] / "checkpoints/best.pt"))
            self.assertEqual(configs["longforcing"]["lineage"]["causal_checkpoint_path"], str(runs["causal"] / "checkpoints/best.pt"))
            self.assertEqual(configs["longforcing"]["rollout"], original["rollout"])
            lineage = configs["eval"]["lineage"]
            self.assertEqual(lineage["expected_stage"], "longforcing_lite_v1")
            self.assertEqual(lineage["checkpoint_path"], str(runs["longforcing"] / "checkpoints/best.pt"))
            self.assertEqual(lineage["artifact_paths"]["training_config"], str(paths["longforcing"]))
            self.assertEqual(lineage["artifact_paths"]["causal_checkpoint"], str(runs["causal"] / "checkpoints/best.pt"))
            self.assertIsNone(lineage["checkpoint_sha256"])
            for scene in configs["eval"]["scenes"]:
                self.assertEqual(Path(scene["initial_frame_path"]).parent, root / "demo_inputs" / scene["scene_id"])
            template = yaml.safe_load((ROOT / generator.TEMPLATES["action"]).read_text(encoding="utf-8"))
            previous_root = str(PurePosixPath(template["model"]["base_model_path"]).parent.parent)
            self.assertNotIn(previous_root, "\n".join(path.read_text(encoding="utf-8") for path in paths.values()))

    def test_all_four_loaders_and_causal_eval_selection(self):
        from training.config import load_config
        from training.causal_tf import load_causal_config
        from training.longforcing_lite import load_longforcing_config
        from training.eval.rollout15s import load_rollout_config

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for stage in ("causal", "longforcing"):
                paths = generator.configure(base / "dataset", base / stage, eval_stage=stage)
                self.assertEqual(load_config(paths["action"]).training.max_steps, 1075)
                self.assertEqual(load_causal_config(paths["causal"]).training.max_steps, 1075)
                self.assertEqual(load_longforcing_config(paths["longforcing"]).training.max_steps, 80)
                evaluation = load_rollout_config(paths["eval"])
                self.assertEqual(evaluation.lineage.artifact_paths["training_config"], str(paths[stage]))
                self.assertEqual("causal_checkpoint" in evaluation.lineage.artifact_paths, stage == "longforcing")

    def test_unsafe_roots_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "new"
            for bad in ("", ".", "relative/data", "~/data", "~", base.anchor, str(Path(base.anchor) / "child" / "..")):
                with self.subTest(root=bad), self.assertRaises(ValueError):
                    generator.configure(bad, output)
                self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                generator.configure(base / "data", output, eval_stage="unknown")
            occupied = base / "already-exists"
            occupied.mkdir()
            marker = occupied / "preserve.txt"
            marker.write_text("existing data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generator.configure(base / "data", occupied)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing data")
            with self.assertRaises(ValueError):
                generator.configure(marker, output)


if __name__ == "__main__":
    unittest.main()
