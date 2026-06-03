import argparse
import datetime
import importlib
import json
import os
import sys
import time
import traceback
import warnings
from functools import partial

import numpy as np
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

import hashlib
import inspect
from pathlib import Path
from typing import Union

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from lmms_eval import evaluator, utils
from lmms_eval.models import get_model
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import (
    make_table,
    simple_parse_args_string,
)

def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        print(f"Argument requires {max_len} integers or None, separated by '{split_char}'. " "Missing values will be filled with defaults.")
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
            else:
                continue


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def _count_generated_tokens(lm, text):
    tokenizer = getattr(lm, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(lm, "_tokenizer", None)
    if tokenizer is None:
        return len(str(text).split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))
    except Exception:
        return len(str(text).split())


def _wrap_generate_until_speed(lm, engine_label: str):
    if getattr(lm, "_qmllm_speed_wrapped", False) or not hasattr(lm, "generate_until"):
        return lm

    original_generate_until = lm.generate_until
    speed_stats = {
        "requests": 0,
        "generated_tokens": 0,
        "generate_sec": 0.0,
    }

    def generate_until_with_speed(requests, *args, **kwargs):
        start = time.perf_counter()
        outputs = original_generate_until(requests, *args, **kwargs)
        speed_stats["generate_sec"] += time.perf_counter() - start
        speed_stats["requests"] += len(requests)
        speed_stats["generated_tokens"] += sum(_count_generated_tokens(lm, text) for text in outputs)

        requests_count = speed_stats["requests"]
        generate_sec = speed_stats["generate_sec"]
        generated_tokens = speed_stats["generated_tokens"]
        samples_per_sec = requests_count / generate_sec if generate_sec > 0 else 0.0
        output_tokens_per_sec = generated_tokens / generate_sec if generate_sec > 0 else 0.0
        avg_output_tokens = generated_tokens / requests_count if requests_count else 0.0
        print(
            f"[{engine_label}_SPEED] "
            f"requests={requests_count} "
            f"generated_tokens={generated_tokens} "
            f"generate_sec={generate_sec:.6f} "
            f"samples_per_sec={samples_per_sec:.6f} "
            f"output_tokens_per_sec={output_tokens_per_sec:.6f} "
            f"avg_output_tokens={avg_output_tokens:.6f} "
            f"batch_size={getattr(lm, 'batch_size', '')}"
        )
        return outputs

    lm.generate_until = generate_until_with_speed
    lm._qmllm_speed_wrapped = True
    return lm


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=32,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument("--process_with_media", action="store_true", help="Whether you will process you dataset with audio, image. By default set to False" "In case some benchmarks need to be processed with media, set this flag to True.")

    # calibration parameters
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", None])
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)

    # TODO: quantization parameters
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq","qig", "rtn", "gptq","igptq", None])
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--a_bit", default=16, type=int)
    parser.add_argument("--w_group", default=128, type=int)
    parser.add_argument("--alpha", default=0.5, type=int)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])
    parser.add_argument("--scale_path", default=None, type=str)
    parser.add_argument("--run_process", action="store_true")
    parser.add_argument("--pseudo_quant", action="store_true")
    parser.add_argument("--real_quant", action="store_true")
    parser.add_argument("--inference_engine", default="hf", choices=["hf", "vllm", "trtllm"])
    parser.add_argument("--vllm_model_path", default=None, type=str)
    parser.add_argument("--vllm_processor_path", default=None, type=str)
    parser.add_argument("--vllm_quantization", default="gptq", type=str)
    parser.add_argument("--vllm_dtype", default="float16", type=str)
    parser.add_argument("--vllm_tensor_parallel_size", default=1, type=int)
    parser.add_argument("--vllm_gpu_memory_utilization", default=0.9, type=float)
    parser.add_argument("--vllm_max_model_len", default=None, type=int)
    parser.add_argument("--vllm_max_num_seqs", default=8, type=int)
    parser.add_argument("--vllm_trust_remote_code", action="store_true")
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--trtllm_model_path", default=None, type=str)
    parser.add_argument("--trtllm_tokenizer_path", default=None, type=str)
    parser.add_argument("--trtllm_model_type", default=None, type=str)
    parser.add_argument("--trtllm_backend", default="engine", choices=["engine", "pytorch"])
    parser.add_argument("--trtllm_dtype", default="auto", type=str)
    parser.add_argument("--trtllm_tensor_parallel_size", default=1, type=int)
    parser.add_argument("--trtllm_pipeline_parallel_size", default=1, type=int)
    parser.add_argument("--trtllm_max_batch_size", default=8, type=int)
    parser.add_argument("--trtllm_max_num_tokens", default=8192, type=int)
    parser.add_argument("--trtllm_max_multimodal_len", default=1296, type=int)
    parser.add_argument("--trtllm_max_seq_len", default=None, type=int)
    parser.add_argument("--trtllm_max_input_len", default=None, type=int)
    parser.add_argument("--trtllm_kv_cache_free_gpu_memory_fraction", default=0.9, type=float)
    parser.add_argument("--trtllm_trust_remote_code", action="store_true")
    parser.add_argument("--trtllm_image_data_format", default="pil", choices=["pil", "pt"])
    parser.add_argument("--trtllm_engine_dir", default=None, type=str)
    parser.add_argument("--trtllm_workspace", default=None, type=str)
    parser.add_argument("--trtllm_enable_build_cache", action="store_true")
    parser.add_argument("--trtllm_fast_build", action="store_true")
    parser.add_argument("--trtllm_scheduler_context_chunking_policy", default=None, type=str)
    # for GPTQ
    parser.add_argument("--percdamp", default=0.01, type=float)

    
    args = parser.parse_args()
    return args


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_eval_args()

    # Check if no arguments were passed after parsing
    if len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                          │")
        print("│ `lmms-eval --model llava --model_path liuhaotian/llava-v1.6-7b --tasks okvqa` │")
        print("│ Use `lmms-eval --help` for more information.                                  │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    if args.wandb_args:
        if "name" not in args.wandb_args:
            name = f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # reset logger
    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args_list = []
    results_list = []
    had_error = False
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator
    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    if accelerator.is_main_process:
        is_main_process = True
    else:
        is_main_process = False

    for args in args_list:
        try:
            # if is_main_process and args.wandb_args:  # thoughtfully we should only init wandb once, instead of multiple ranks to avoid network traffics and unwanted behaviors.
            #     wandb_logger = WandbLogger()
            print(f"Evaluating with args: {args}")
            results, samples = cli_evaluate_single(args)
            results_list.append(results)

            accelerator.wait_for_everyone()
            if is_main_process and args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to Weights and Biases failed due to {e}")
                # wandb_logger.finish()

        except Exception as e:
            had_error = True
            if args.verbosity == "DEBUG":
                raise e
            else:
                traceback.print_exc()
                eval_logger.error(f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
                results_list.append(None)

    for args, results in zip(args_list, results_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            print(f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, " f"batch_size: {args.batch_size}")
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

    if args.wandb_args:
        wandb_logger.run.finish()

    if had_error:
        sys.exit(1)


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    selected_task_list = args.tasks.split(",") if args.tasks else None

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

    # update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"

    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    eval_logger.info(f"Evaluation tracker args: {evaluation_tracker_args}")

    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("If fewshot_as_multiturn is set, apply_chat_template must be set to True.")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub.")

    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        args.include_path = [args.include_path] if args.include_path else []
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            args.include_path.append(package_tasks_location)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format(f"\n - ".join(sorted(task_manager.list_all_tasks()))))
        sys.exit()
    elif args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [task for task in task_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n" f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    eval_logger.info(f"Selected Tasks: {task_names}")
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    # here we load MLLMs outside of the evaluator.
    if args.model_args is None:
        args.model_args = ""
    
    if args.real_quant and args.inference_engine == "trtllm":
        if not args.trtllm_model_path:
            raise ValueError("--real_quant --inference_engine trtllm requires --trtllm_model_path")
        if not args.trtllm_tokenizer_path:
            raise ValueError(
                "--real_quant --inference_engine trtllm requires --trtllm_tokenizer_path "
                "pointing to the original HF VLM checkpoint."
            )
        if args.w_bit == 3:
            raise ValueError(
                "TensorRT-LLM does not currently advertise W3A16 engine support. "
                "Use --w_bit 4 with --a_bit 16/8 for TRT-LLM, or keep W3A16 on the documented vLLM fallback."
            )
        if not (args.w_bit == 4 and args.a_bit in (8, 16)):
            raise ValueError("The TensorRT-LLM real-quant path is wired for W4A16 and W4A8.")
        from qmllm.trtllm_eval import TRTLLMRealQuantModel

        lm = TRTLLMRealQuantModel(
            pretrained=args.trtllm_model_path,
            tokenizer_path=args.trtllm_tokenizer_path,
            model_type=args.trtllm_model_type,
            backend=args.trtllm_backend,
            dtype=args.trtllm_dtype,
            tensor_parallel_size=args.trtllm_tensor_parallel_size,
            pipeline_parallel_size=args.trtllm_pipeline_parallel_size,
            max_batch_size=args.trtllm_max_batch_size,
            max_num_tokens=args.trtllm_max_num_tokens,
            max_multimodal_len=args.trtllm_max_multimodal_len,
            max_seq_len=args.trtllm_max_seq_len,
            max_input_len=args.trtllm_max_input_len,
            kv_cache_free_gpu_memory_fraction=args.trtllm_kv_cache_free_gpu_memory_fraction,
            trust_remote_code=args.trtllm_trust_remote_code,
            image_data_format=args.trtllm_image_data_format,
            engine_dir=args.trtllm_engine_dir,
            workspace=args.trtllm_workspace,
            enable_build_cache=args.trtllm_enable_build_cache,
            fast_build=args.trtllm_fast_build,
            scheduler_context_chunking_policy=args.trtllm_scheduler_context_chunking_policy,
            batch_size=args.batch_size,
        )
    elif args.real_quant and args.inference_engine == "vllm":
        if args.model != "qwen2_vl":
            raise ValueError("The real-quant vLLM eval path currently supports --model qwen2_vl")
        if not args.vllm_model_path:
            raise ValueError("--real_quant --inference_engine vllm requires --vllm_model_path")
        from qmllm.vllm_eval import VLLMRealQuantQwen2VL

        lm = VLLMRealQuantQwen2VL(
            pretrained=args.vllm_model_path,
            processor_path=args.vllm_processor_path,
            quantization=args.vllm_quantization,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            trust_remote_code=args.vllm_trust_remote_code,
            enforce_eager=args.vllm_enforce_eager,
            batch_size=args.batch_size,
        )
    else:
        ModelClass = get_model(args.model)
        lm = ModelClass.create_from_arg_string(
            args.model_args,
            {
                "batch_size": args.batch_size,
                "device": args.device,
            },
        )
    # 
    if args.pseudo_quant:
        from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
        from qmllm.calibration.pileval import get_calib_dataset
        from qmllm.models import get_process_model
        from qmllm.quantization.quant_wrapper import qwrapper

        print("Pseudo quant...")
        Process_ModelClass = get_process_model(args.model)
        process_model = Process_ModelClass(lm._model, 
                                        lm._tokenizer, 
                                        lm.processor if hasattr(lm, 'processor') else None)


        prompt_inputs = None
        prompt_kwargs = None
        if args.method in ["gptq","igptq"]:
            if args.calib_data == "pileval":
                prompt_inputs, prompt_kwargs = get_calib_dataset(data_path=args.data_path, tokenizer=lm._tokenizer, n_samples=args.n_samples)
            elif args.calib_data == "coco":
                prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(data_path=args.data_path,
                                                                            image_folder=args.image_folder,
                                                                            model=process_model,
                                                                            n_samples=args.n_samples,
                                                                            few_shot_format=args.few_shot_format,
                                                                            interleave_format=args.interleave_format,
                                                                            text_data_path=args.text_data_path)

        

        qwrapper(process_model, prompt_inputs, prompt_kwargs, args)
        _wrap_generate_until_speed(lm, "FAKE_QUANT")

    evaluate_kwargs = {
        "model_args": args.model_args,
        "tasks": task_names,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size,
        "device": args.device,
        "use_cache": args.use_cache,
        "limit": args.limit,
        "check_integrity": args.check_integrity,
        "write_out": args.write_out,
        "log_samples": args.log_samples,
        "evaluation_tracker": evaluation_tracker,
        "system_instruction": args.system_instruction,
        "apply_chat_template": args.apply_chat_template,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "gen_kwargs": args.gen_kwargs,
        "task_manager": task_manager,
        "verbosity": args.verbosity,
        "predict_only": args.predict_only,
        "random_seed": args.seed[0],
        "numpy_random_seed": args.seed[1],
        "torch_random_seed": args.seed[2],
        "fewshot_random_seed": args.seed[3],
        "cli_args": args,
        "datetime_str": datetime_str,
        **request_caching_args,
    }
    if "lm" in inspect.signature(evaluator.simple_evaluate).parameters:
        evaluate_kwargs.update({"model": args.model, "lm": lm})
    else:
        evaluate_kwargs.update({"model": lm})

    results = evaluator.simple_evaluate(**evaluate_kwargs)

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None, datetime_str=datetime_str)

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        return results, samples
    return None, None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}),\ngen_kwargs: ({args.gen_kwargs}),\nlimit: {args.limit},\nnum_fewshot: {args.num_fewshot},\nbatch_size: {args.batch_size}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()
