# OpenVINO multi-model workflow

This notebook processes a video, detects people, finds and compares their faces, and runs gesture recognition only for the matched person. Models and sample media are downloaded into `models/` and `data/` on the first run.

The included Open Model Zoo classroom video provides a small crowd demonstration. Set `INPUT_VIDEO` and `REFERENCE_IMAGE` in the notebook to use your own video and target photo.

#### Linux / macOS:
```bash
./launch.sh
```

#### Windows:
```cmd
launch.bat
```
