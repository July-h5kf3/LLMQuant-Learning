from transformers import AddedToken, LlamaTokenizer


def load_lisa_tokenizer(
    model_path,
    *,
    model_max_length=None,
    padding_side="right",
    use_fast=False,
    **kwargs,
):
    return LlamaTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        **kwargs,
    )


def add_lisa_seg_token(tokenizer):
    num_added_tokens = tokenizer.add_tokens(
        [AddedToken("[SEG]", lstrip=True, normalized=False)]
    )
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    return num_added_tokens, seg_token_id
