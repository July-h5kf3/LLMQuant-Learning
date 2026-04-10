import os

from tabulate import tabulate


def get_exp_name(args):
    exp_name = getattr(args, "exp_name", None)
    if exp_name:
        return exp_name

    model_name = os.path.basename(os.path.normpath(args.version))
    dataset_name = args.test_dataset.replace("|", "_")
    return f"{model_name}_{dataset_name}"


def get_precision_label(args):
    if args.quant_method == "awq":
        return f"{args.precision}+{args.quant_kwargs.get('bits', 4)}bit"
    if args.quant_method == "bnb_4bit":
        return f"{args.precision}+4bit"
    if args.quant_method == "bnb_8bit":
        return f"{args.precision}+8bit"
    if args.quant_method == "gptq":
        return f"{args.precision}+{args.quant_kwargs.get('bits', 4)}bit"
    if args.quant_method == "hqq":
        return f"{args.precision}+{args.quant_kwargs.get('bits', 4)}bit"
    if args.quant_method == "quanto":
        return f"{args.precision}+{args.quant_kwargs.get('weights', 'int4')}"
    if args.quant_method == "smoothquant":
        return (
            f"{args.precision}+w{args.quant_kwargs.get('w_bit', 4)}"
            f"a{args.quant_kwargs.get('a_bit', 8)}"
        )
    return f"{args.precision}+{args.precision}"


def get_method_label(args):
    if args.quant_method == "awq":
        return "awq"
    if args.quant_method == "bnb_4bit":
        quant_type = args.quant_kwargs.get("quant_type", "nf4")
        use_double_quant = args.quant_kwargs.get("use_double_quant", True)
        suffix = "+double_quant" if use_double_quant else ""
        return f"bnb_{quant_type}{suffix}"
    if args.quant_method == "bnb_8bit":
        return "bnb_llm.int8"
    if args.quant_method == "gptq":
        return "gptq"
    if args.quant_method == "hqq":
        return "hqq"
    if args.quant_method == "quanto":
        return f"quanto_{args.quant_kwargs.get('weights', 'int4')}"
    if args.quant_method == "smoothquant":
        return "smoothquant"
    return "none"


def get_report_path(args):
    report_path = getattr(args, "report_path", None)
    if report_path:
        return report_path
    filename = f"{get_exp_name(args)}.md"
    return os.path.join(os.path.dirname(__file__), "results", filename)


def write_markdown_result(args, metrics):
    report_path = get_report_path(args)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    if "mAP" in metrics:
        headers = [
            "Exp_name",
            "Precision",
            "Method",
            "mAP",
            "AP50",
            "AP75",
            "AP-small",
            "AP-medium",
            "AP-large",
            "mem_peak",
            "avg_fwd",
        ]
        values = [
            get_exp_name(args),
            get_precision_label(args),
            get_method_label(args),
            f"{metrics['mAP']:.4f}" if metrics.get("mAP") is not None else "NA",
            f"{metrics['AP50']:.4f}" if metrics.get("AP50") is not None else "NA",
            f"{metrics['AP75']:.4f}" if metrics.get("AP75") is not None else "NA",
            f"{metrics['AP-small']:.4f}" if metrics.get("AP-small") is not None else "NA",
            f"{metrics['AP-medium']:.4f}" if metrics.get("AP-medium") is not None else "NA",
            f"{metrics['AP-large']:.4f}" if metrics.get("AP-large") is not None else "NA",
            f"{metrics['peak_mem_mib']:.2f} MiB" if metrics.get("peak_mem_mib") is not None else "NA",
            f"{metrics['avg_fwd_ms']:.2f}" if metrics.get("avg_fwd_ms") is not None else "NA",
        ]
    else:
        headers = [
            "Exp_name",
            "Precision",
            "Method",
            "cIOU",
            "gIOU",
            "mem_peak",
            "avg_fwd",
        ]
        values = [
            get_exp_name(args),
            get_precision_label(args),
            get_method_label(args),
            f"{metrics['ciou']:.4f}" if metrics.get("ciou") is not None else "NA",
            f"{metrics['giou']:.4f}" if metrics.get("giou") is not None else "NA",
            f"{metrics['peak_mem_mib']:.2f} MiB" if metrics.get("peak_mem_mib") is not None else "NA",
            f"{metrics['avg_fwd_ms']:.2f}" if metrics.get("avg_fwd_ms") is not None else "NA",
        ]
    table = tabulate([values], headers=headers, tablefmt="github")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print(f"Markdown report saved to: {report_path}")
    return report_path
