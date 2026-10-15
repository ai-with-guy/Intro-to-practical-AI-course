# OpenVINO face search

This beginner-friendly notebook searches a video for a reference face using just two models: face detection and face embeddings. There is no body detection or gesture recognition. Models and sample media are downloaded into `models/` and `data/` on the first run.

Run `openvino_workflow.ipynb` from top to bottom. The included classroom video provides a ready-to-run demo; a face from its opening frames becomes the reference. Set `INPUT_VIDEO` and `REFERENCE_IMAGE` to use your own video and target photo, with permission from the people involved.

The core search is a short `search_faces` function: **detect → crop → embed → compare**. Face crops are embedded in one model batch, and `embeddings @ reference_vector` compares all faces at once. The notebook explains preprocessing and cosine similarity step by step. `video_utils.py` keeps video I/O and drawing out of the model code.

Green boxes indicate similarity above `MATCH_THRESHOLD`; gray boxes indicate scores below it. Scores are not identity probabilities. The annotated video is saved to `data/openvino_result.webm`. Simple face crops keep the lesson short, but real identity applications need alignment, threshold calibration, testing, and privacy safeguards.

The final notebook section uses OpenVINO's `benchmark_app` with separate Throughput and Latency panels, following the layout in [This is my AI PC](https://github.com/guy-tamir/gen-ai-pc/tree/main/2). Select a model and device, run, and compare devices in the plot. Each panel shows the command used. Both downloaded models have FP16 weights. `benchmark_utils.py` handles the UI; all timing is done by `benchmark_app`. Synthetic inputs and batch size one keep the comparison model-only, not end-to-end video FPS. You can also run the tool from an activated environment:

```bash
benchmark_app -m models/face-detection-retail-0005/face-detection-retail-0005.xml -d CPU -hint latency -b 1 -t 10
```

Try `-d GPU` or `-d NPU` if available, or `-hint throughput` to measure concurrent inference. Use `benchmark_app -h` to explore the options.

#### Linux / macOS:
```bash
./launch.sh
```

#### Windows:
```cmd
launch.bat
```
