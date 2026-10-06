"""Replay the generated prefix alongside the attention used for each prediction."""

from .head_view import head_view


def generation_view(example, *, layer=None, heads=None, include_layers=None,
                    prettify_tokens=True, html_action='view'):
    """Display an inspect_translation result, without running the model again.

    Each step reads a prefix, reveals its recorded next-token prediction, then
    appends that token. Decoder attention is cropped to the available prefix;
    cross-attention keeps the full encoder input. The final appended token has
    no query row because it was not fed back into the recorded model pass.
    """
    return head_view(
        encoder_attention=example.get('encoder_attention'),
        decoder_attention=example.get('decoder_attention'),
        cross_attention=example.get('cross_attention'),
        encoder_tokens=example['encoder_tokens'],
        decoder_tokens=example['decoder_tokens'],
        generation_tokens=example['predicted_tokens'],
        layer=layer, heads=heads, include_layers=include_layers,
        prettify_tokens=prettify_tokens, html_action=html_action)
