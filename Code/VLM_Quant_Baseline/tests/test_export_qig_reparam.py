import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import TestCase, main


def load_export_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "export_qig_reparam_under_test",
        root / "export_qig_reparam.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QIGReparamExportTest(TestCase):
    def setUp(self):
        self.module = load_export_module()

    def test_apply_qig_to_model_requires_non_empty_scale(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            self.module.apply_qig_to_model(object(), {}, apply_fn=lambda *_: None)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.module.apply_qig_to_model(object(), {"scale": []}, apply_fn=lambda *_: None)

    def test_apply_qig_to_model_invokes_apply_fn_with_full_model(self):
        model = object()
        qig_results = {"scale": [("model.layers.0.input_layernorm", ["model.layers.0.self_attn.q_proj"], [1.0])]}
        calls = []

        returned = self.module.apply_qig_to_model(
            model,
            qig_results,
            apply_fn=lambda received_model, received_results: calls.append(
                (received_model, received_results)
            ),
        )

        self.assertIs(returned, model)
        self.assertEqual(calls, [(model, qig_results)])

    def test_write_manifest_records_qig_source_and_scale_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                model="qwen2_vl",
                model_dir="/weights/Qwen2-VL-7B-Instruct",
                scale_path="/scale/QIG/qig/qwen2_vl_7b_w4a8/qwen2_vl_7b_w4a8_qig.pt",
                output_dir=tmpdir,
                w_bit=4,
                a_bit=8,
            )
            self.module.write_manifest(args, {"scale": ["s0", "s1"]})

            manifest = json.loads((Path(tmpdir) / "qig_reparam_manifest.json").read_text())

        self.assertEqual(manifest["source_model_dir"], args.model_dir)
        self.assertEqual(manifest["scale_path"], args.scale_path)
        self.assertEqual(manifest["backend_input_kind"], "hf_qig_reparameterized_full_precision")
        self.assertEqual(manifest["qig_num_scales"], 2)
        self.assertEqual(manifest["qig_applied_to"], "full_huggingface_model")

    def test_ensure_empty_or_forced_refuses_non_empty_output_without_force(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out"
            out.mkdir()
            (out / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                self.module.ensure_empty_or_forced(out, force=False)

            self.module.ensure_empty_or_forced(out, force=True)
            self.assertTrue(out.exists())
            self.assertFalse(any(out.iterdir()))


if __name__ == "__main__":
    main()
