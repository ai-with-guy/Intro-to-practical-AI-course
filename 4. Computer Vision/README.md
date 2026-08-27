# Computer Vision

This lesson downloads MNIST and CIFAR-10 into `data/` the first time the
notebook runs. CIFAR-10 is about 170 MB and MNIST is about 50 MB.

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
`pyproject.toml`. Training will also work on CPU, but the CIFAR-10 section will
take longer.
