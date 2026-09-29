"""Notebook-friendly checks for the lesson's attention_math exercise.

Usage in bert_lesson.ipynb:
    from attention_math_tests import run_attention_math_tests
    run_attention_math_tests(attention_math)
"""

from html import escape
import os
import sys

import torch
import torch.nn.functional as F


def _report(message, *, success):
    """Show one colored result in a notebook, or print it in a terminal."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None and get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            color = "#176b3a" if success else "#a12b2b"
            background = "#eaf7ee" if success else "#fff0ef"
            display(HTML(
                f'<div style="color:{color};background:{background};'
                f'border-left:4px solid {color};padding:10px 14px;white-space:pre-wrap">'
                f'{escape(message)}</div>'
            ))
            return
    except ImportError:
        pass

    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        color = "\033[32m" if success else "\033[31m"
        print(f"{color}{message}\033[0m")
    else:
        print(message)


def _check_output(fn, q, k, v, scale):
    expected = F.scaled_dot_product_attention(q, k, v, scale=scale)
    actual = fn(q, k, v, scale)
    if not isinstance(actual, torch.Tensor):
        return "Expected a torch.Tensor as the return value."
    if actual.shape != expected.shape:
        return f"Expected output shape {tuple(expected.shape)}, got {tuple(actual.shape)}."
    if not torch.isfinite(actual).all():
        return "The output contains NaN or infinity; check numerical stability."
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-7):
        return "The output values do not match the expected attention result."
    return None


def _check_gradients(fn, q, k, v, scale):
    inputs = [tensor.clone().detach().requires_grad_() for tensor in (q, k, v)]
    reference_inputs = [tensor.clone().detach().requires_grad_() for tensor in (q, k, v)]
    expected = F.scaled_dot_product_attention(*reference_inputs, scale=scale)
    actual = fn(*inputs, scale)
    if not isinstance(actual, torch.Tensor):
        return "Expected a torch.Tensor as the return value."
    if actual.shape != expected.shape:
        return f"Expected output shape {tuple(expected.shape)}, got {tuple(actual.shape)}."
    if not actual.requires_grad:
        return "The output is detached from the inputs; training needs gradients."
    if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, rtol=1e-5, atol=1e-7):
        return "The output values do not match the expected attention result."
    expected.square().sum().backward()
    actual.square().sum().backward()
    for name, tensor, reference in zip(("q", "k", "v"), inputs, reference_inputs):
        if tensor.grad is None:
            return f"No gradient reached {name}; check that the computation stays in PyTorch."
        if not torch.allclose(tensor.grad, reference.grad, rtol=1e-5, atol=1e-7):
            return f"The gradient for {name} is incorrect; check the forward computation."
    return None


def run_attention_math_tests(fn):
    """Show the first problem with fn(q, k, v, scale); return True iff all pass."""
    if not callable(fn):
        _report("Not a function: pass attention_math itself, not its result.", success=False)
        return False

    rng = torch.Generator().manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float64)

    cases = [
        (
            "Basic attention",
            (rand(3, 4), rand(3, 4), rand(3, 4), 0.5),
            "Check the dimensions of q and k when computing scores: which dimensions need to align?",
        ),
        (
            "Different query, key, and value sizes",
            (rand(2, 3), rand(5, 3), rand(5, 2), 3 ** -0.5),
            "Queries and keys can have different sequence lengths; values can have a different feature size.",
        ),
        (
            "Batch dimension",
            (rand(3, 2, 4), rand(3, 5, 4), rand(3, 5, 2), 0.5),
            "Each batch item should be processed independently; check which dimensions you use.",
        ),
        (
            "Extra leading dimensions",
            (rand(2, 3, 4, 5), rand(2, 3, 6, 5), rand(2, 3, 6, 2), 5 ** -0.5),
            "The attention calculation should also work with multiple leading dimensions.",
        ),
        (
            "Zero scale",
            (rand(2, 3), rand(4, 3), rand(4, 2), 0.0),
            "A scale of zero should give every key equal weight for each query.",
        ),
        (
            "Negative scale",
            (rand(3, 3), rand(4, 3), rand(4, 2), -0.7),
            "Use the supplied scale as given, including its sign.",
        ),
        (
            "Large scores",
            (rand(2, 3) * 1000, rand(4, 3) * 1000, rand(4, 2), 1.0),
            "Large scores should still produce finite, correctly weighted values.",
        ),
    ]

    total = len(cases) + 1
    for number, (name, args, hint) in enumerate(cases, start=1):
        try:
            problem = _check_output(fn, *args)
        except Exception as exc:
            problem = f"Raised {type(exc).__name__}: {exc}"
        if problem is not None:
            _report(f"Check {number}/{total} - {name}: {problem}\nHint: {hint}", success=False)
            return False

    try:
        problem = _check_gradients(fn, rand(2, 3, 4), rand(2, 5, 4), rand(2, 5, 3), 0.5)
    except Exception as exc:
        problem = f"Raised {type(exc).__name__}: {exc}"
    if problem is not None:
        _report(f"Check {total}/{total} - Gradients for training: {problem}\n"
                "Hint: Keep the computation differentiable.", success=False)
        return False

    _report(f"All {total} checks passed!", success=True)
    return True
