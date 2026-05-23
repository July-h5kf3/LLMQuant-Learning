import base64
import os
from io import BytesIO
from typing import List, Optional, Union

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms


class VLLMRealQuantQwen2VL(lmms):
    def __init__(
        self,
        pretrained: str,
        processor_path: Optional[str] = None,
        quantization: str = "gptq",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 8,
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs for VLLMRealQuantQwen2VL: {kwargs}")

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm import LLM

        self.model_path = pretrained
        self.processor_path = processor_path or pretrained
        self.batch_size_per_gpu = int(batch_size)
        self._rank = 0
        self._world_size = 1

        self.processor = AutoProcessor.from_pretrained(self.processor_path, trust_remote_code=trust_remote_code)
        self._tokenizer = AutoTokenizer.from_pretrained(self.processor_path, trust_remote_code=trust_remote_code)
        self.processor.tokenizer.padding_side = "left"
        self._tokenizer.padding_side = "left"

        llm_kwargs = {
            "model": pretrained,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
        }
        if quantization:
            llm_kwargs["quantization"] = quantization
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("Loglikelihood is not implemented for VLLMRealQuantQwen2VL")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation is not implemented for VLLMRealQuantQwen2VL")

    def _flatten_visuals(self, visual):
        if visual is None:
            return []
        if isinstance(visual, Image.Image):
            return [visual]
        if isinstance(visual, (list, tuple)):
            flattened = []
            for item in visual:
                flattened.extend(self._flatten_visuals(item))
            return flattened
        return [visual]

    def _visual_to_content(self, visual):
        if isinstance(visual, Image.Image):
            buffer = BytesIO()
            visual.convert("RGB").save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return {"type": "image", "image": f"data:image/jpeg;base64,{encoded}"}, visual.convert("RGB")
        return {"type": "image", "image": str(visual)}, visual

    def _build_request(self, context, doc_to_visual, doc_id, task, split):
        context = context.replace("<image>", "").strip()
        doc = self.task_dict[task][split][doc_id]
        visuals = self._flatten_visuals(doc_to_visual(doc))

        content = []
        mm_images = []
        for visual in visuals:
            visual_content, mm_image = self._visual_to_content(visual)
            content.append(visual_content)
            mm_images.append(mm_image)
        content.append({"type": "text", "text": context})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        request = {"prompt": prompt}
        if mm_images:
            request["multi_modal_data"] = {"image": mm_images if len(mm_images) > 1 else mm_images[0]}
        return request

    def _generate(self, requests, sampling_params):
        try:
            return self.llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
        except TypeError as exc:
            if "use_tqdm" not in str(exc):
                raise
            return self.llm.generate(requests, sampling_params=sampling_params)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        from vllm import SamplingParams

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="VLLM Real-Quant Responding")
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

            sampling_kwargs = {
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }
            if top_p is not None:
                sampling_kwargs["top_p"] = top_p
            if until:
                sampling_kwargs["stop"] = until
            sampling_params = SamplingParams(**sampling_kwargs)

            vllm_requests = [
                self._build_request(context, doc_to_visual, doc_id, task, split)
                for context, doc_to_visual, doc_id, task, split in zip(contexts, doc_to_visuals, doc_ids, tasks, splits)
            ]
            outputs = self._generate(vllm_requests, sampling_params)

            for output, context in zip(outputs, contexts):
                ans = output.outputs[0].text.strip() if output.outputs else ""
                if ans.lower().startswith("assistant"):
                    ans = ans[len("assistant"):].lstrip(":： \n\t")
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), ans)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
