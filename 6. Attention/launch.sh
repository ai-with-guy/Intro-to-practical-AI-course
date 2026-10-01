#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo "Launching your local Jupyter environment via uv..."
torch_extra="${TORCH_EXTRA:-cuda}"

if [[ -z "${TORCH_EXTRA:-}" && "$(uname -s)" == "Linux" ]]; then
    intel_gpu=false
    nvidia_gpu=false
    for vendor_file in /sys/class/drm/card*/device/vendor; do
        [[ -r "$vendor_file" ]] || continue
        case "$(<"$vendor_file")" in
            0x8086) intel_gpu=true ;;
            0x10de) nvidia_gpu=true ;;
        esac
    done
    if [[ "$intel_gpu" == true && "$nvidia_gpu" == false ]]; then
        torch_extra="xpu"
    fi
fi

if [[ "$torch_extra" == "xpu" ]]; then
    echo "Using the PyTorch XPU build."
fi
uv run --exact --extra "$torch_extra" jupyter lab "attention.ipynb"
