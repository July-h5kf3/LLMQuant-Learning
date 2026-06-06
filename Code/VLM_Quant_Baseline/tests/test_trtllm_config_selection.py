import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import TestCase, main


def load_trtllm_eval_module():
    root = Path(__file__).resolve().parents[1]

    class _DummyImage:
        pass

    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.Image = _DummyImage
    pil_module.Image = image_module
    sys.modules.setdefault("PIL", pil_module)
    sys.modules.setdefault("PIL.Image", image_module)

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda *args, **kwargs: None
    sys.modules.setdefault("tqdm", tqdm_module)

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoConfig = object
    transformers_module.AutoProcessor = object
    transformers_module.AutoTokenizer = object
    sys.modules.setdefault("transformers", transformers_module)

    lmms_eval_module = types.ModuleType("lmms_eval")
    lmms_eval_module.utils = types.SimpleNamespace()
    api_module = types.ModuleType("lmms_eval.api")
    instance_module = types.ModuleType("lmms_eval.api.instance")
    instance_module.Instance = object
    model_module = types.ModuleType("lmms_eval.api.model")
    model_module.lmms = object
    sys.modules.setdefault("lmms_eval", lmms_eval_module)
    sys.modules.setdefault("lmms_eval.api", api_module)
    sys.modules.setdefault("lmms_eval.api.instance", instance_module)
    sys.modules.setdefault("lmms_eval.api.model", model_module)

    spec = importlib.util.spec_from_file_location(
        "trtllm_eval_under_test",
        root / "qmllm" / "trtllm_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TRTLLMConfigSelectionTest(TestCase):
    def setUp(self):
        self.module = load_trtllm_eval_module()

    def write_config(self, path, payload):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_qwen2vl_engine_backend_uses_engine_llm_config_for_safety_check(self):
        tmp = Path(self._testMethodName)
        checkpoint = tmp / "checkpoint"
        engine_llm = tmp / "engine" / "llm"
        try:
            self.write_config(
                checkpoint,
                {
                    "model_type": "qwen2",
                    "architectures": ["Qwen2ForCausalLM"],
                },
            )
            self.write_config(
                engine_llm,
                {
                    "pretrained_config": {
                        "model_type": "qwen2_vl",
                        "architectures": ["Qwen2VLForConditionalGeneration"],
                        "position_embedding_type": "mrope",
                    }
                },
            )

            config = self.module._select_trtllm_runtime_config(
                pretrained=str(checkpoint),
                engine_dir=str(tmp / "engine"),
                backend="engine",
                model_type="qwen2_vl",
            )

            self.assertEqual(config["model_type"], "qwen2_vl")
            self.assertEqual(config["architectures"], ["Qwen2VLForConditionalGeneration"])
        finally:
            if tmp.exists():
                for child in sorted(tmp.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    else:
                        child.rmdir()
                tmp.rmdir()


if __name__ == "__main__":
    main()
