#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo 'Launching your local Jupyter environment via uv...'
torch_extra="${TORCH_EXTRA:-cuda}"

if [[ -z "${TORCH_EXTRA:-}" ]]; then
    case "$(uname -s)" in
        Linux)
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
            ;;
        MINGW*|MSYS*|CYGWIN*)
            detected="$(powershell.exe -NoProfile -Command '$gpus = Get-CimInstance Win32_VideoController; if (($gpus.AdapterCompatibility -match "Intel") -and -not ($gpus.AdapterCompatibility -match "NVIDIA")) { "xpu" }')"
            if [[ "${detected//$'\r'/}" == "xpu" ]]; then
                torch_extra="xpu"
            fi
            ;;
    esac
fi

if [[ "$torch_extra" == "xpu" ]]; then
    echo 'Using the PyTorch XPU build.'
fi
exec uv run --exact --extra "$torch_extra" jupyter lab Building_a_Bash_Agent.ipynb "$@"
