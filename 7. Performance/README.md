# Class 7: Performance

- `training_memory.ipynb`: PyTorch memory, tensor lifetimes, and inference mode.
- `openvino_workflow.ipynb`: OpenVINO face search and device benchmarks.
- `gguf_quantization.ipynb`: Load FP16 and Q4_K_M GGUF files directly with OpenVINO GenAI on the Intel GPU; compare file size and generation speed.

Run `launch.bat` on Windows or `bash launch.sh` on Linux/macOS. The launchers install the shared dependencies with uv and select the PyTorch XPU build for Intel-only GPU systems.

Or launch explicitly:

```sh
uv run --extra xpu jupyter lab
# NVIDIA/default PyTorch:
uv run --extra cuda jupyter lab
```

The GGUF notebook uses Qwen3-1.7B: the current GenAI GGUF loader supports Qwen3, not Qwen3.5. Its matching files total about 5.4 GB and are cached on first download. No BF16 model, llama-cpp-python, conversion, or notebook installation cells. Model inference explicitly targets `GPU`.

The PyTorch notebook deliberately demonstrates out-of-memory errors; run it cell by cell. The face-search notebook downloads sample media and models on first run.
