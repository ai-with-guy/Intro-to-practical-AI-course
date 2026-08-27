#!/usr/bin/env bash

echo "Launching your local Jupyter environment via uv..."

torch_extra="cuda"

if [[ "$(uname -s)" == "Linux" ]]; then
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
        echo "Intel-only GPU system detected; using the PyTorch XPU build."
        torch_extra="xpu"
    fi
fi

uv run --exact --extra "$torch_extra" jupyter lab "transformer.ipynb"
