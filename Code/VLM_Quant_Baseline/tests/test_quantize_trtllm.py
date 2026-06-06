import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch


def load_quantize_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quantize_trtllm_under_test",
        root / "quantize_trtllm.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class W3AutoRoundExportTest(TestCase):
    def setUp(self):
        self.module = load_quantize_module()

    def _args(self, output_dir):
        return argparse.Namespace(
            model="qwen2_vl",
            model_dir="/models/qwen2-vl-qig-reparam-w3a16",
            output_dir=output_dir,
            quant_format="w3a16",
            tp_size=1,
            pp_size=1,
            cp_size=1,
            autoround_format="auto_round",
            autoround_dataset="/data/qig_coco_train2017_caption.json",
            autoround_iters=10,
            autoround_nsamples=8,
            autoround_seqlen=512,
            autoround_batch_size=1,
            autoround_device_map="0",
            autoround_template="qwen2_vl",
            autoround_extra_args="--extra_data_dir /data/images",
        )

    def test_w3_export_uses_available_auto_round_mllm_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(tmpdir)

            with patch.object(self.module.subprocess, "run") as run:
                self.module.export_w3_autoround(args)

            cmd = run.call_args.args[0]
            self.assertEqual(cmd[0], "auto-round")
            self.assertIn("--mllm", cmd)
            self.assertNotIn("auto-round-mllm", cmd)
            self.assertIn("--template", cmd)
            self.assertIn("qwen2_vl", cmd)
            self.assertIn("--extra_data_dir", cmd)

            manifest = json.loads((Path(tmpdir) / "vlm_quant_real_manifest.json").read_text())

        self.assertEqual(manifest["backend"], "autoround-w3-fallback")
        self.assertEqual(manifest["source_model_dir"], args.model_dir)
        self.assertEqual(manifest["autoround_dataset"], args.autoround_dataset)


if __name__ == "__main__":
    main()
