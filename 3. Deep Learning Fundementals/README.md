#### Linux / macOS:
```bash
./launch.sh
```

#### Windows:
```cmd
launch.bat
```

The launchers select `cuda` when an NVIDIA GPU is present. On systems with Intel
graphics and no NVIDIA GPU, they select the `xpu` dependency extra from
`pyproject.toml`. The XPU extra uses PyTorch's official XPU wheel index and
requires a supported Intel GPU and current graphics drivers.
