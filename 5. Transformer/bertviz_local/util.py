import torch


def _attention_layers(attention):
    """Distinguish a single tensor/array or nested matrix from a layer list."""
    if hasattr(attention, 'shape'):
        return [attention]
    if not isinstance(attention, (list, tuple)) or not attention:
        raise ValueError("Attention must be a tensor, array, or non-empty list of layers.")
    if not hasattr(attention[0], 'shape'):
        try:
            if torch.as_tensor(attention).ndim == 2:
                return [attention]
        except (TypeError, ValueError, RuntimeError):
            pass
    return attention


def format_attention(attention, layers=None, heads=None):
    """Return [layers, heads, queries, keys], adding missing singleton axes.

    Each layer accepts [queries, keys], [heads, queries, keys], or
    [1, heads, queries, keys]. A [1, queries, keys] single-head batch
    also works. Select one example before passing a larger batch.
    """
    attention = _attention_layers(attention)
    if layers is not None:
        if not layers:
            raise ValueError("Select at least one attention layer.")
        if any(index < 0 or index >= len(attention) for index in layers):
            raise ValueError(f"Layer indices must be between 0 and {len(attention) - 1}.")
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        layer_attention = torch.as_tensor(layer_attention).detach().cpu()
        if layer_attention.ndim == 2:
            layer_attention = layer_attention.unsqueeze(0)
        elif layer_attention.ndim == 4:
            if layer_attention.shape[0] != 1:
                raise ValueError("Attention batch size must be 1. Select one example before visualization.")
            layer_attention = layer_attention.squeeze(0)
        elif layer_attention.ndim != 3:
            raise ValueError("Each attention layer must have shape [queries, keys], "
                             "[heads, queries, keys], or [1, heads, queries, keys].")
        if any(size == 0 for size in layer_attention.shape):
            raise ValueError("Attention must contain at least one head, query, and key.")
        if heads is not None:
            if not heads or any(index < 0 or index >= layer_attention.shape[0] for index in heads):
                raise ValueError("Select valid, non-empty attention head indices.")
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    if any(tensor.shape != squeezed[0].shape for tensor in squeezed):
        raise ValueError("All attention layers must have the same number of heads, queries, and keys.")
    return torch.stack(squeezed)


def num_layers(attention):
    return len(_attention_layers(attention))


def num_heads(attention):
    return format_attention(attention, layers=[0]).shape[1]


def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]
