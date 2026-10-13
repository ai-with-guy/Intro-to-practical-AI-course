# OpenVINO face search

This beginner-friendly notebook searches a video for a reference face using just two models: face detection and face embeddings. There is no body detection or gesture recognition. Models and sample media are downloaded into `models/` and `data/` on the first run.

Run `openvino_workflow.ipynb` from top to bottom. The included classroom video provides a ready-to-run demo; a face from its opening frames becomes the reference. Set `INPUT_VIDEO` and `REFERENCE_IMAGE` to use your own video and target photo, with permission from the people involved.

The core search is a short `search_faces` function: **detect → crop → embed → compare**. Face crops are embedded in one model batch, and `embeddings @ reference_vector` compares all faces at once. The notebook explains preprocessing and cosine similarity step by step. `video_utils.py` keeps video I/O and drawing out of the model code.

Green boxes indicate similarity above `MATCH_THRESHOLD`; gray boxes indicate scores below it. Scores are not identity probabilities. The annotated video is saved to `data/openvino_result.webm`. Simple face crops keep the lesson short, but real identity applications need alignment, threshold calibration, testing, and privacy safeguards.

The final notebook section benchmarks both models separately on available CPU, GPU, and NPU devices. It shows batch-one latency and concurrent inference throughput in a table and charts. Compilation and preprocessing are excluded from timing. Unsupported model/device combinations are skipped with an explanation. `benchmark_utils.py` contains the timing helpers.

#### Linux / macOS:
```bash
./launch.sh
```

#### Windows:
```cmd
launch.bat
```
