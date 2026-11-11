# From model tokens to a Bash agent

Double-click **`launch.bat`** on Windows, or run `bash Agents/launch.sh` from
the repository root. Like the other course projects, the launcher uses
`uv run --exact --extra ...` to sync the project and open
**`Building_a_Bash_Agent.ipynb`** in its local environment. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and Git for Windows
first. No Ollama, Docker, model server, or separate Jupyter setup is needed.

## What students build

1. Start with text between `<bash>` tags and pass the body unchanged to Bash.
2. Parse **raw decoded model tokens**, including the end-of-message marker,
   into a command or final answer. Reject malformed/incomplete output.
3. Format Bash observations back into Qwen3's token-framed conversation.
4. Repeat inference to create an agent, then optionally add a JSON action API.

The notebook shows Hugging Face **tokenizer → `model.generate()` → decode**
directly. There is no pipeline, chat server, built-in tool parser, or hidden
output cleanup. `skip_special_tokens=False` keeps the model's boundary tokens
visible. We supply the Qwen3 framing rather than expect it to be memorized.
Native tool-call markers are control tokens, not ordinary prose; the notebook
compares their IDs with our teaching delimiters and shows explicit stopping
logic. It also shows Qwen3's thinking/non-thinking prompt prefixes: an empty
reasoning block is prefilled for this exercise, not stripped from the output.

## Model and hardware

The first model-loading cell downloads **`Qwen/Qwen3-0.6B`** from Hugging Face
and caches it locally. Allow roughly 1–2 GB for model downloads plus several
GB of working RAM and environment space. A public model needs no login.
Subsequent loads use the cache; Python dependencies also require internet on
first launch. All recorded-response exercises work without loading a model.

`pyproject.toml` declares **Python >=3.11,<3.14**, project version `0.1.0`,
notebook/Transformers dependencies, and mutually exclusive **`cuda` / `xpu`**
PyTorch extras with the same indexes as `6. Attention`. `uv.lock` pins the
resolved environment; there are no parallel requirements files.

The launchers select the XPU build for Intel-only systems and the standard
PyTorch build otherwise. Windows uses GPU detection via PowerShell; Linux
uses DRM vendor IDs. Git Bash also supports Windows detection. Override before
launch with `$env:TORCH_EXTRA = 'cuda'` or `'xpu'` (PowerShell), or
`TORCH_EXTRA=xpu bash Agents/launch.sh` (Bash).

The notebook chooses available CUDA, Intel XPU, or Apple MPS, falling back to
CPU, and prints the actual device. GPU detection selects a **build**, not a
guarantee of driver/device support. Float32 keeps the initial example simple;
you may explicitly set `DEVICE = 'cpu'`. This uses PyTorch, not Vulkan.
Try `Qwen/Qwen3-1.7B` if the tiny model follows the protocol poorly and you
have enough memory.

The hand-written template is **Qwen3-specific**. Changing model families also
requires changing the chat framing and tokenizer assumptions. The source is
[Qwen3's tokenizer configuration/template](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json).

## Files worth keeping

- `Building_a_Bash_Agent.ipynb`: the lesson, with exercise solutions included during development.
- `workspace/`: tiny disposable dataset.
- `launch.bat`, `launch.sh`, `pyproject.toml`, `uv.lock`: course-style environment.

The notebook is self-contained, including the supplied Bash runtime, solutions,
and scripted checks. No helper-module imports or separate test file are needed.

## Bash behavior and safety

`bash(command)` asks permission, runs the exact string in a fresh Git Bash
shell in the workspace, and returns stdout, stderr and exit code. File changes
persist; `cd` and exports do not. It selects Git Bash, not Windows' WSL launcher.
For a custom installation set `GIT_BASH` to its `bin/bash.exe` before launching.

This is **not a sandbox**. Commands have your host permissions and inherited
environment, can leave the workspace and access the network. Review every
command and use only non-sensitive sample data. Use a disposable VM for
untrusted experiments. Output clipping is after capture, and a timeout does
not guarantee descendant processes die. Strong isolation is a separate topic.
