"""Empty lmms-eval task plugin namespace.

Some lmms-eval versions use LMMS_EVAL_PLUGINS for both model and task plugins.
Our package registers only a model wrapper, but exposing this namespace lets
those versions resolve prune_quant_baseline.lmms_eval.tasks safely.
"""
