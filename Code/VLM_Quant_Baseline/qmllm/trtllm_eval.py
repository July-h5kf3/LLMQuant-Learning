import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms


def _coerce_batch_size(batch_size: Optional[Union[int, str]]) -> int:
    if batch_size is None:
        return 1
    if isinstance(batch_size, int):
        return max(1, batch_size)
    value = str(batch_size).strip().lower()
    if value.startswith("auto"):
        return 1
    return max(1, int(value))


def _load_model_type(path: str, override: Optional[str] = None) -> str:
    if override:
        return override
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if not model_type:
        raise ValueError(
            "Cannot infer TensorRT-LLM multimodal model_type from config. "
            "Pass --trtllm_model_type explicitly."
        )
    return model_type


def _strip_image_tokens(text: str) -> str:
    return text.replace("<image>", " ").strip()


MODEL_TYPE_ALIASES = {
    "qwen3_moe_vl": "qwen3_vl_moe",
    "llava_onevision": "llava_next",
    "vila": "llava_llama",
}

SUPPORTED_MODELS = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "llava_next",
    "llava_llama",
}


def _canonical_model_type(model_type: str) -> str:
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


def _model_type_supports_default_loader(model_type: str) -> bool:
    return _canonical_model_type(model_type) in SUPPORTED_MODELS


def _call_with_supported_kwargs(fn, **kwargs):
    parameters = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return fn(**kwargs)
    return fn(**{key: value for key, value in kwargs.items() if key in parameters})


def _load_trtllm_checkpoint_config(path: str) -> Dict[str, Any]:
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


class TRTLLMRealQuantModel(lmms):
    """lmms-eval adapter for TensorRT-LLM real-quant multimodal generation."""

    def __init__(
        self,
        pretrained: str,
        tokenizer_path: Optional[str] = None,
        model_type: Optional[str] = None,
        backend: str = "engine",
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_batch_size: int = 8,
        max_num_tokens: int = 8192,
        max_multimodal_len: int = 1296,
        max_seq_len: Optional[int] = None,
        max_input_len: Optional[int] = None,
        kv_cache_free_gpu_memory_fraction: float = 0.9,
        trust_remote_code: bool = True,
        image_data_format: str = "pil",
        engine_dir: Optional[str] = None,
        workspace: Optional[str] = None,
        enable_build_cache: bool = False,
        fast_build: bool = False,
        scheduler_context_chunking_policy: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs for TRTLLMRealQuantModel: {kwargs}")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRT_LLM_NO_LIB_INIT", "1")
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "8.9")

        self.model_path = pretrained
        self.tokenizer_path = tokenizer_path or pretrained
        inferred_model_type = _load_model_type(self.tokenizer_path, model_type)
        self.model_type = _canonical_model_type(inferred_model_type)
        self._checkpoint_config = _load_trtllm_checkpoint_config(pretrained)
        self._checkpoint_model_type = self._checkpoint_config.get("model_type")
        self._checkpoint_architectures = self._checkpoint_config.get("architectures") or []
        if (
            self.model_type == "qwen2_vl"
            and self._checkpoint_model_type == "qwen2"
            and "Qwen2ForCausalLM" in self._checkpoint_architectures
        ):
            raise ValueError(
                "This TensorRT-LLM checkpoint is configured as plain qwen2/Qwen2ForCausalLM, "
                "but the tokenizer_path is Qwen2-VL. That combination silently produces "
                "garbage because Qwen2-VL requires mRoPE and multimodal prompt-tuning inputs. "
                "Re-export the checkpoint with qwen2_vl/mRoPE config using quantize_trtllm.py."
            )
        if not _model_type_supports_default_loader(self.model_type):
            raise ValueError(
                f"TensorRT-LLM multimodal input support for model_type={self.model_type!r} "
                "is not enabled in this adapter. Start with qwen2_vl/qwen2_5_vl, "
                "qwen3_vl, llava_next/llava_onevision, or vila."
            )
        self.backend = backend
        self.image_data_format = image_data_format
        self.batch_size_per_gpu = _coerce_batch_size(batch_size)
        self._rank = 0
        self._world_size = 1
        self._speed_stats = {
            "requests": 0,
            "generated_tokens": 0,
            "generate_sec": 0.0,
        }
        self._qwen2vl_runner = None
        self._qwen2vl_image_size = 504

        self._processor = AutoProcessor.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        self._tokenizer.padding_side = "left"
        if getattr(self._processor, "tokenizer", None) is not None:
            self._processor.tokenizer.padding_side = "left"

        try:
            import tensorrt_llm._torch.models  # noqa: F401
            from tensorrt_llm.builder import BuildConfig
            from tensorrt_llm.plugin import _load_plugin_lib
            from tensorrt_llm.llmapi import LLM as TorchLLM
            from tensorrt_llm.sampling_params import SamplingParams
            from tensorrt_llm.inputs import create_input_processor
            from tensorrt_llm.llmapi import KvCacheConfig
            from tensorrt_llm.llmapi import SchedulerConfig
            from tensorrt_llm.llmapi import ContextChunkingPolicy
            from tensorrt_llm.inputs import default_multimodal_input_loader
            from tensorrt_llm.runtime.multimodal_model_runner import MultimodalModelRunner
            from tensorrt_llm.tokenizer import TransformersTokenizer
        except ImportError as exc:
            raise ImportError(
                "TensorRT-LLM is required for --inference_engine trtllm. "
                "Install it in the target NVIDIA environment with `pip install tensorrt_llm`."
            ) from exc

        try:
            _load_plugin_lib()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the TensorRT-LLM plugin library. W4A8 AWQ needs "
                "TensorRT plugin creators such as WeightOnlyGroupwiseQuantMatmul; "
                "make sure tensorrt_llm/libs is visible in LD_LIBRARY_PATH."
            ) from exc

        trt_tokenizer = TransformersTokenizer(self._tokenizer)
        self.SamplingParams = SamplingParams
        self.default_multimodal_input_loader = default_multimodal_input_loader
        self.create_input_processor = create_input_processor

        if backend not in {"engine", "pytorch"}:
            raise ValueError("--trtllm_backend must be 'engine' or 'pytorch'")

        if backend == "engine" and self.model_type == "qwen2_vl":
            self._qwen2vl_runner = self._init_qwen2vl_multimodal_runner(
                MultimodalModelRunner,
                engine_dir=engine_dir,
                max_batch_size=max_batch_size,
                kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
                trust_remote_code=trust_remote_code,
                scheduler_context_chunking_policy=scheduler_context_chunking_policy,
            )
            return

        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        )
        scheduler_config = None
        if scheduler_context_chunking_policy:
            try:
                context_chunking_policy = ContextChunkingPolicy[scheduler_context_chunking_policy]
            except KeyError as exc:
                valid = ", ".join(policy.name for policy in ContextChunkingPolicy)
                raise ValueError(
                    "--trtllm_scheduler_context_chunking_policy must be one of "
                    f"{valid}, got {scheduler_context_chunking_policy!r}."
                ) from exc
            scheduler_config = SchedulerConfig(
                context_chunking_policy=context_chunking_policy,
            )
        llm_kwargs: Dict[str, Any] = {
            "model": pretrained,
            "tokenizer": trt_tokenizer,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "dtype": dtype,
            "kv_cache_config": kv_cache_config,
        }
        if scheduler_config is not None:
            llm_kwargs["scheduler_config"] = scheduler_config
        if backend == "engine":
            from tensorrt_llm._tensorrt_engine import LLM

            build_config_kwargs = {
                "max_batch_size": max_batch_size,
                "max_num_tokens": max_num_tokens,
                "max_prompt_embedding_table_size": max_multimodal_len,
            }
            if max_seq_len is not None:
                build_config_kwargs["max_seq_len"] = max_seq_len
            if max_input_len is not None:
                build_config_kwargs["max_input_len"] = max_input_len
            build_config = BuildConfig(**build_config_kwargs)
            llm_kwargs["build_config"] = build_config
            if workspace:
                llm_kwargs["workspace"] = workspace
            if enable_build_cache:
                llm_kwargs["enable_build_cache"] = True
            if fast_build:
                llm_kwargs["fast_build"] = True
        else:
            LLM = TorchLLM
            llm_kwargs["backend"] = "pytorch"
            llm_kwargs["max_batch_size"] = max_batch_size
            llm_kwargs["max_num_tokens"] = max_num_tokens
            if max_seq_len is not None:
                llm_kwargs["max_seq_len"] = max_seq_len
            if max_input_len is not None:
                llm_kwargs["max_input_len"] = max_input_len

        self.llm = LLM(**llm_kwargs)
        self._patch_multimodal_input_processor(trust_remote_code)
        if engine_dir and hasattr(self.llm, "save"):
            self.llm.save(engine_dir)

    def _init_qwen2vl_multimodal_runner(
        self,
        runner_cls,
        *,
        engine_dir: Optional[str],
        max_batch_size: int,
        kv_cache_free_gpu_memory_fraction: float,
        trust_remote_code: bool,
        scheduler_context_chunking_policy: Optional[str],
    ):
        if scheduler_context_chunking_policy:
            raise ValueError(
                "TensorRT-LLM MultimodalModelRunner does not expose the LLM API "
                "scheduler_context_chunking_policy knob. Leave "
                "--trtllm_scheduler_context_chunking_policy unset for the Qwen2-VL engine path."
            )
        if not engine_dir:
            raise ValueError(
                "Qwen2-VL TensorRT engine backend requires --trtllm_engine_dir pointing "
                "to a prebuilt multimodal engine directory with vision/ and llm/ subdirs. "
                "The old LLM API path is disabled because it omits Qwen2-VL mRoPE and "
                "prompt-table inputs, which produces fast but incorrect garbage outputs."
            )

        engine_path = Path(engine_dir)
        _require_file(engine_path / "vision" / "config.json", "Qwen2-VL vision engine config")
        _require_file(engine_path / "vision" / "model.engine", "Qwen2-VL vision engine")
        _require_file(engine_path / "llm" / "config.json", "Qwen2-VL LLM engine config")
        if not list((engine_path / "llm").glob("rank*.engine")):
            raise FileNotFoundError(f"Missing Qwen2-VL LLM rank engine under {engine_path / 'llm'}")

        self.batch_size_per_gpu = max(1, min(self.batch_size_per_gpu, max_batch_size))
        args = SimpleNamespace(
            engine_dir=str(engine_path),
            hf_model_dir=self.tokenizer_path,
            batch_size=self.batch_size_per_gpu,
            session="cpp_llm_only",
            visual_engine_name="model.engine",
            image_path=None,
            video_path=None,
            video_num_frames=None,
            path_sep=",",
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            num_beams=1,
            lora_task_uids=None,
            enable_chunked_context=False,
            mm_embedding_offloading=None,
            enable_context_fmha_fp32_acc=False,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=None,
            multi_block_mode=True,
            debug_mode=False,
            trust_remote_code=trust_remote_code,
        )
        runner = runner_cls(args)
        self.llm = runner
        self._tokenizer = runner.tokenizer
        self._tokenizer.padding_side = "right"
        if getattr(self._processor, "tokenizer", None) is not None:
            self._processor.tokenizer = runner.tokenizer
        return runner

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def _patch_multimodal_input_processor(self, trust_remote_code: bool) -> None:
        """Point TRT-LLM multimodal preprocessing at the original HF processor dir."""
        if not hasattr(self.llm, "input_processor"):
            return
        try:
            self.llm.input_processor = _call_with_supported_kwargs(
                self.create_input_processor,
                model_path_or_dir=self.tokenizer_path,
                tokenizer=self.llm.tokenizer,
                trust_remote_code=trust_remote_code,
            )
            self.llm._hf_model_dir = Path(self.tokenizer_path)
            self.llm._tokenizer = self.llm.input_processor.tokenizer
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize TensorRT-LLM multimodal input processor "
                f"from tokenizer_path={self.tokenizer_path!r}. For VLMs, "
                "--trtllm_tokenizer_path must point to the original HF checkpoint."
            ) from exc

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("Loglikelihood is not implemented for TRTLLMRealQuantModel")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation is not implemented for TRTLLMRealQuantModel")

    def _flatten_visuals(self, visual):
        if visual is None:
            return []
        if isinstance(visual, Image.Image):
            return [visual.convert("RGB")]
        if isinstance(visual, (list, tuple)):
            flattened = []
            for item in visual:
                flattened.extend(self._flatten_visuals(item))
            return flattened
        if isinstance(visual, (str, os.PathLike)):
            return [str(visual)]
        return [visual]

    def _to_qwen2vl_runner_image(self, visual: Any) -> Image.Image:
        if isinstance(visual, Image.Image):
            image = visual.convert("RGB")
        elif isinstance(visual, (str, os.PathLike)):
            image = Image.open(visual).convert("RGB")
        else:
            raise TypeError(
                "Qwen2-VL TensorRT runner expects PIL images or image paths from lmms-eval, "
                f"got {type(visual)!r}."
            )
        return image.resize((self._qwen2vl_image_size, self._qwen2vl_image_size))

    def _build_qwen2vl_runner_request(self, context, doc_to_visual, doc_id, task, split):
        doc = self.task_dict[task][split][doc_id]
        visuals = self._flatten_visuals(doc_to_visual(doc))
        if not visuals:
            raise ValueError(
                "The Qwen2-VL TensorRT multimodal runner path currently expects one image per request. "
                "Use the PyTorch/debug backend for pure-text checks."
            )
        if len(visuals) != 1:
            raise ValueError(
                "The Qwen2-VL TensorRT multimodal runner batch path currently expects exactly "
                f"one image per request, got {len(visuals)}. Run multi-image tasks with batch_size=1 "
                "after adding a dedicated multi-image prompt-table layout."
            )
        return _strip_image_tokens(context), self._to_qwen2vl_runner_image(visuals[0])

    def _apply_text_chat_template(self, prompt: str) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def _load_multimodal_prompt(self, prompt: str, visuals: List[Any]) -> Dict[str, Any]:
        modality = "multiple_image" if len(visuals) > 1 else "image"
        return _call_with_supported_kwargs(
            self.default_multimodal_input_loader,
            tokenizer=self.tokenizer,
            model_dir=self.tokenizer_path,
            model_type=self.model_type,
            modality=modality,
            prompts=[prompt],
            media=[visuals],
            image_data_format=self.image_data_format,
            device="cpu",
            trust_remote_code=True,
        )[0]

    def _build_request(self, context, doc_to_visual, doc_id, task, split):
        doc = self.task_dict[task][split][doc_id]
        visuals = self._flatten_visuals(doc_to_visual(doc))
        prompt = _strip_image_tokens(context)

        if not visuals:
            return {"prompt": self._apply_text_chat_template(prompt)}

        return self._load_multimodal_prompt(prompt, visuals)

    def _generate(self, requests, sampling_params):
        try:
            return self.llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
        except TypeError as exc:
            if "use_tqdm" not in str(exc):
                raise
            return self.llm.generate(requests, sampling_params=sampling_params)

    def _normalize_runner_sampling(self, temperature, top_p, top_k) -> Dict[str, Any]:
        temperature = 1.0 if temperature is None else float(temperature)
        if temperature <= 0:
            temperature = 1.0
            top_k = 1
        return {
            "temperature": temperature,
            "top_p": 1.0 if top_p is None else float(top_p),
            "top_k": 1 if top_k is None else int(top_k),
        }

    def _generate_qwen2vl_runner(
        self,
        prompts: List[str],
        images: List[Image.Image],
        *,
        max_new_tokens: int,
        temperature,
        top_p,
        top_k,
    ) -> List[str]:
        runner = self._qwen2vl_runner
        if runner is None:
            raise RuntimeError("Qwen2-VL TensorRT multimodal runner was not initialized.")

        sampling = self._normalize_runner_sampling(temperature, top_p, top_k)
        runner.args.batch_size = len(prompts)
        runner.args.temperature = sampling["temperature"]
        runner.args.top_p = sampling["top_p"]
        runner.args.top_k = sampling["top_k"]

        (
            _input_text,
            pre_prompt,
            post_prompt,
            image,
            decoder_input_ids,
            other_vision_inputs,
            other_audio_inputs,
            other_decoder_inputs,
        ) = runner.setup_inputs(prompts, images)
        outputs = runner.generate(
            pre_prompt,
            post_prompt,
            image,
            decoder_input_ids,
            max_new_tokens,
            other_vision_inputs=other_vision_inputs,
            other_audio_inputs=other_audio_inputs,
            other_decoder_inputs=other_decoder_inputs,
        )
        return [beam_group[0].strip() if beam_group else "" for beam_group in outputs]

    def _count_generated_tokens(self, output, text: str) -> int:
        if output.outputs:
            token_ids = getattr(output.outputs[0], "token_ids", None)
            if token_ids is not None:
                return len(token_ids)
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(self.tokenizer.encode(text))

    def _print_speed_summary(self) -> None:
        requests = self._speed_stats["requests"]
        generate_sec = self._speed_stats["generate_sec"]
        generated_tokens = self._speed_stats["generated_tokens"]
        samples_per_sec = requests / generate_sec if generate_sec > 0 else 0.0
        output_tokens_per_sec = generated_tokens / generate_sec if generate_sec > 0 else 0.0
        avg_output_tokens = generated_tokens / requests if requests else 0.0
        print(
            "[TRTLLM_SPEED] "
            f"requests={requests} "
            f"generated_tokens={generated_tokens} "
            f"generate_sec={generate_sec:.6f} "
            f"samples_per_sec={samples_per_sec:.6f} "
            f"output_tokens_per_sec={output_tokens_per_sec:.6f} "
            f"avg_output_tokens={avg_output_tokens:.6f} "
            f"batch_size={self.batch_size}"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="TRT-LLM Real-Quant Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        res = []

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits = zip(*chunk)
            gen_kwargs = dict(all_gen_kwargs[0])

            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected until to be str or list, got {type(until)}")

            max_new_tokens = gen_kwargs.pop("max_new_tokens", 128)
            temperature = gen_kwargs.pop("temperature", 0)
            top_p = gen_kwargs.pop("top_p", None)
            top_k = gen_kwargs.pop("top_k", None)

            sampling_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }
            if top_p is not None:
                sampling_kwargs["top_p"] = top_p
            if top_k is not None:
                sampling_kwargs["top_k"] = top_k
            if until:
                sampling_kwargs["stop"] = until

            start = time.perf_counter()
            if self._qwen2vl_runner is not None:
                runner_requests = [
                    self._build_qwen2vl_runner_request(context, doc_to_visual, doc_id, task, split)
                    for context, doc_to_visual, doc_id, task, split in zip(contexts, doc_to_visuals, doc_ids, tasks, splits)
                ]
                prompts, images = zip(*runner_requests)
                outputs = self._generate_qwen2vl_runner(
                    list(prompts),
                    list(images),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                sampling_params = self.SamplingParams(**sampling_kwargs)
                trtllm_requests = [
                    self._build_request(context, doc_to_visual, doc_id, task, split)
                    for context, doc_to_visual, doc_id, task, split in zip(contexts, doc_to_visuals, doc_ids, tasks, splits)
                ]
                outputs = self._generate(trtllm_requests, sampling_params)
            self._speed_stats["generate_sec"] += time.perf_counter() - start

            for output, context in zip(outputs, contexts):
                ans = output if isinstance(output, str) else (output.outputs[0].text.strip() if output.outputs else "")
                if ans.lower().startswith("assistant"):
                    ans = ans[len("assistant"):].lstrip(":： \n\t")
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                self._speed_stats["requests"] += 1
                if isinstance(output, str):
                    self._speed_stats["generated_tokens"] += len(self.tokenizer.encode(ans, add_special_tokens=False))
                else:
                    self._speed_stats["generated_tokens"] += self._count_generated_tokens(output, ans)
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), ans)
                pbar.update(1)

        pbar.close()
        if self.rank == 0:
            self._print_speed_summary()
        return re_ords.get_original(res)
