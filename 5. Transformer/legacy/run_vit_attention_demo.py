"""Open the interactive ViT attention demo in Matplotlib.

Examples:
    uv run --extra cuda python run_vit_attention_demo.py
    uv run --extra cuda python run_vit_attention_demo.py --image "Golf ball"
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import RadioButtons, Slider
from PIL import Image
from tqdm.auto import tqdm

from vit_attention_utils import (
    HEATMAP_RENDER_OPTIONS,
    IMAGENET_EXAMPLES,
    MODEL_OPTIONS,
    _attention_image,
    _best_device,
    _grid_image,
    load_vit_model,
    run_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        choices=IMAGENET_EXAMPLES,
        default=next(iter(IMAGENET_EXAMPLES)),
        help="Bundled ImageNet example to display (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_OPTIONS.values(),
        default="vit_b_16",
        help="torchvision model name (default: %(default)s).",
    )
    parser.add_argument("--device", help="For example: cpu, cuda, or xpu.")
    return parser.parse_args()


def show_attention(result, image_name: str) -> None:
    selected_patch = [result.grid_size[0] // 2, result.grid_size[1] // 2]
    figure, (source_axis, attention_axis) = plt.subplots(1, 2, figsize=(12, 6))
    figure.subplots_adjust(left=0.04, right=0.78, bottom=0.2, top=0.9, wspace=0.08)

    source_artist = source_axis.imshow(
        _grid_image(result.image, result.grid_size, tuple(selected_patch))
    )
    attention_artist = attention_axis.imshow(
        _attention_image(
            result,
            layer=0,
            head=None,
            selected_patch=tuple(selected_patch),
            opacity=0.7,
            render_mode="grayscale",
        )
    )
    source_axis.set_title(f"{image_name}: click a query patch")
    attention_axis.set_title(
        f"{result.prediction} ({result.confidence:.1%}) | mean attention"
    )
    source_axis.set_axis_off()
    attention_axis.set_axis_off()

    layer_axis = figure.add_axes((0.12, 0.1, 0.48, 0.04))
    strength_axis = figure.add_axes((0.12, 0.04, 0.48, 0.04))
    mode_axis = figure.add_axes((0.81, 0.32, 0.18, 0.22))
    layer_slider = Slider(
        layer_axis,
        "Layer",
        1,
        len(result.attention),
        valinit=1,
        valstep=1,
    )
    strength_slider = Slider(
        strength_axis,
        "Overlay",
        0.1,
        1.0,
        valinit=0.7,
        valstep=0.05,
    )
    mode_picker = RadioButtons(mode_axis, list(HEATMAP_RENDER_OPTIONS))
    mode_axis.set_title("Render")

    def redraw(_event=None) -> None:
        selected = tuple(selected_patch)
        source_artist.set_data(_grid_image(result.image, result.grid_size, selected))
        attention_artist.set_data(
            _attention_image(
                result,
                layer=int(layer_slider.val) - 1,
                head=None,
                selected_patch=selected,
                opacity=float(strength_slider.val),
                render_mode=HEATMAP_RENDER_OPTIONS[mode_picker.value_selected],
            )
        )
        figure.canvas.draw_idle()

    def select_patch(event) -> None:
        if event.inaxes is not source_axis or event.xdata is None or event.ydata is None:
            return
        rows, columns = result.grid_size
        selected_patch[1] = min(int(event.xdata / result.image.width * columns), columns - 1)
        selected_patch[0] = min(int(event.ydata / result.image.height * rows), rows - 1)
        redraw()

    layer_slider.on_changed(redraw)
    strength_slider.on_changed(redraw)
    mode_picker.on_clicked(redraw)
    figure.canvas.mpl_connect("button_press_event", select_patch)
    plt.show()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else _best_device()
    download_bar: tqdm | None = None

    def update_download(downloaded: int, total: int | None) -> None:
        nonlocal download_bar
        if download_bar is None:
            download_bar = tqdm(
                total=total,
                desc="Model weights",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )
        download_bar.update(downloaded - download_bar.n)

    print(f"Loading {args.model} on {device}...")
    try:
        model, weights = load_vit_model(args.model, device, update_download)
    finally:
        if download_bar is not None:
            download_bar.close()

    with Image.open(IMAGENET_EXAMPLES[args.image]) as image:
        result = run_attention(model, weights, image, device)
    print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
    show_attention(result, args.image)


if __name__ == "__main__":
    main()
