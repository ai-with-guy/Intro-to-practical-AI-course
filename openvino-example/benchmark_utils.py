"""A notebook-friendly UI around OpenVINO's benchmark_app."""

from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import gradio as gr
import matplotlib.pyplot as plt


MODELS = {
    'Face detection': 'face-detection-retail-0005',
    'Face embedding': 'face-reidentification-retail-0095',
}
COLORS = {'CPU': 'tab:blue', 'GPU': 'tab:green', 'NPU': 'tab:orange'}


def parse_metrics(output):
    """Read the two summary metrics, not compilation or first-inference time."""
    latency = re.search(r'^\[ INFO \]\s+Median:\s+([\d.]+)\s+ms', output, re.MULTILINE)
    throughput = re.search(r'^\[ INFO \]\s+Throughput:\s+([\d.]+)\s+FPS', output, re.MULTILINE)
    if latency is None or throughput is None:
        raise ValueError('Could not read benchmark summary metrics.')
    return float(latency[1]), float(throughput[1])


def plot_results(rows, model, goal):
    """Compare devices for the selected model and optimization goal only."""
    selected = [row for row in rows if row[0] == model and row[2] == goal]
    metric = 3 if goal == 'latency' else 4
    label = 'Latency (ms)' if goal == 'latency' else 'Throughput (FPS)'
    figure, axis = plt.subplots()
    if selected:
        axis.scatter(
            [row[1] for row in selected], [row[metric] for row in selected],
            marker='s', color='tab:orange', label='FP16',
        )
        axis.legend(loc='best')
    axis.set_title(f'{model} {goal.title()} by Device')
    axis.set_ylabel(label)
    axis.set_xlabel('Device')
    figure.tight_layout()
    plt.close(figure)
    return figure


def run_benchmark(model_dir, model, device, goal, seconds):
    """Delegate all warm-up and measurement to the official CLI."""
    executable = shutil.which('benchmark_app', path=str(Path(sys.executable).parent))
    if executable is None:
        raise RuntimeError('Install openvino in the notebook environment to get benchmark_app.')
    name = MODELS[model]
    arguments = ['-m', str(Path(model_dir) / name / f'{name}.xml'),
                 '-d', device, '-hint', goal, '-b', '1', '-t', str(int(seconds))]
    command = (subprocess.list2cmdline(['benchmark_app', *arguments])
               if sys.platform == 'win32' else shlex.join(['benchmark_app', *arguments]))
    result = subprocess.run([executable, *arguments], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return result.returncode, result.stdout, command


def build_benchmark_ui(available_devices, goal, model_dir='models'):
    """Use the reference notebook's separate latency/throughput panel layout."""
    if goal not in {'Latency', 'Throughput'}:
        raise ValueError('goal must be Latency or Throughput')
    hint = goal.lower()
    devices = [device for device in available_devices
               if device.split('.')[0] in COLORS]
    with gr.Blocks() as demo:
        gr.Markdown(f'# Benchmark Model {goal} by Device')
        results = gr.State([])
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                model = gr.Dropdown(list(MODELS), value='Face detection', label='Choose a model')
                precision = gr.Dropdown(['FP16'], value='FP16', label='Choose a precision')
                device = gr.Dropdown(devices, value=devices[0] if devices else None,
                                     label='Choose a device')
                run = gr.Button('Run Benchmark', variant='primary', interactive=bool(devices))
                clear = gr.Button('Clear Plot Data')
            with gr.Column(scale=2, min_width=300):
                plot = gr.Plot()
        with gr.Row():
            command = gr.Textbox(label='Command used for this run:', interactive=False)

        def measure(model, precision, device, rows):
            try:
                returncode, output, used_command = run_benchmark(model_dir, model, device, hint, 30)
                if returncode:
                    raise RuntimeError(output[-2000:])
                latency, throughput = parse_metrics(output)
                rows = [row for row in rows if row[:3] != [model, device, hint]]
                rows = [*rows, [model, device, hint, latency, throughput]]
            except (OSError, RuntimeError, ValueError) as error:
                raise gr.Error(str(error)) from error
            return rows, plot_results(rows, model, hint), used_command

        run.click(measure, inputs=[model, precision, device, results],
                  outputs=[results, plot, command], concurrency_limit=1)
        model.change(lambda rows, model: plot_results(rows, model, hint),
                     inputs=[results, model], outputs=plot, queue=False)

        def clear_results(model):
            return [], plot_results([], model, hint)

        clear.click(clear_results, inputs=model, outputs=[results, plot])
    return demo
