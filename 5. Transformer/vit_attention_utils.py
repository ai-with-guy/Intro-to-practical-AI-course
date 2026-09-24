"""Interactive attention visualization for torchvision Vision Transformers.

The public entry point is ``vit_attention_demo``. The implementation lives in
this module so that the accompanying teaching notebook can stay concise.
"""

from __future__ import annotations

import io
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlparse

import ipywidgets as widgets
import numpy as np
import torch
from IPython.display import display
from ipyevents import Event
from PIL import Image, ImageDraw, ImageOps
from torchvision.models import get_model, get_model_weights


MODEL_OPTIONS = {
    "ViT-B/16 (14 x 14 patches)": "vit_b_16",
    "ViT-B/32 (7 x 7 patches)": "vit_b_32",
}
HEATMAP_RENDER_OPTIONS = {
    "Grayscale": "grayscale",
    "Color": "color",
}
ASSET_DIR = Path(__file__).resolve().parent / "assets"
# Validation images from Imagenette, the 10-class subset of ImageNet.
IMAGENET_EXAMPLES = {
    "Parachute": ASSET_DIR / "parachute.jpeg",
    "English springer": ASSET_DIR / "english-springer.jpeg",
    "Church": ASSET_DIR / "church.jpeg",
    "Golf ball": ASSET_DIR / "golf-ball.jpeg",
}
DEFAULT_IMAGE = next(iter(IMAGENET_EXAMPLES.values()))
DownloadProgress = Callable[[int, int | None], None]


@dataclass
class AttentionRun:
    """Outputs needed to redraw the visualization without rerunning the ViT."""

    image: Image.Image
    attention: tuple[torch.Tensor, ...]
    grid_size: tuple[int, int]
    prediction: str
    confidence: float


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_weights_downloaded(weights, progress: DownloadProgress | None = None) -> None:
    """Cache torchvision weights while exposing progress to non-terminal UIs."""

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint = checkpoint_dir / Path(urlparse(weights.url).path).name
    if checkpoint.exists():
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(f"{checkpoint.suffix}.part")
    downloaded = 0
    try:
        with urllib.request.urlopen(weights.url) as response, partial.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length else None
            if progress is not None:
                progress(0, total)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)
        partial.replace(checkpoint)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def load_vit_model(
    model_name: str,
    device: torch.device | str | None = None,
    download_progress: DownloadProgress | None = None,
) -> tuple[torch.nn.Module, object]:
    """Load a pretrained torchvision ViT, reporting its initial download."""

    device = torch.device(device) if device is not None else _best_device()
    weights = get_model_weights(model_name).DEFAULT
    _ensure_weights_downloaded(weights, download_progress)
    model = get_model(model_name, weights=weights).to(device).eval()
    return model, weights


def _encoder_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    try:
        blocks = list(model.encoder.layers)
    except AttributeError as error:
        raise TypeError(
            "This visualizer expects a torchvision VisionTransformer with "
            "model.encoder.layers."
        ) from error

    if not blocks or any(not hasattr(block, "self_attention") for block in blocks):
        raise TypeError("The model does not expose torchvision ViT attention blocks.")
    return blocks


@contextmanager
def _capture_attention(model: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    """Temporarily ask each MultiheadAttention block for per-head weights."""

    captured: list[torch.Tensor | None] = [None] * len(_encoder_blocks(model))
    originals = []

    for layer_index, block in enumerate(_encoder_blocks(model)):
        attention = block.self_attention
        original_forward = attention.forward
        originals.append((attention, original_forward))

        def forward_with_weights(
            *args,
            _forward=original_forward,
            _layer_index=layer_index,
            **kwargs,
        ):
            # torchvision's EncoderBlock normally sets need_weights=False.
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            output, weights = _forward(*args, **kwargs)
            if weights is None:
                raise RuntimeError("The attention module did not return its weights.")
            captured[_layer_index] = weights.detach().float().cpu()
            return output, weights

        attention.forward = forward_with_weights

    try:
        yield captured
    finally:
        for attention, original_forward in originals:
            attention.forward = original_forward


def _tensor_to_image(tensor: torch.Tensor, transform) -> Image.Image:
    mean = torch.tensor(transform.mean, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(transform.std, dtype=tensor.dtype)[:, None, None]
    pixels = (tensor.cpu() * std + mean).clamp(0, 1)
    array = (pixels.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


@torch.inference_mode()
def run_attention(
    model: torch.nn.Module,
    weights,
    image: Image.Image,
    device: torch.device | str | None = None,
) -> AttentionRun:
    """Run a torchvision ViT once and collect attention from every layer."""

    device = torch.device(device) if device is not None else next(model.parameters()).device
    transform = weights.transforms()
    input_tensor = transform(image.convert("RGB"))
    shown_image = _tensor_to_image(input_tensor, transform)
    batch = input_tensor.unsqueeze(0).to(device)

    model.eval()
    with _capture_attention(model) as captured:
        logits = model(batch)

    if any(layer is None for layer in captured):
        raise RuntimeError("Not all encoder layers produced attention weights.")

    patch_size = model.patch_size
    patch_height = patch_size[0] if isinstance(patch_size, tuple) else patch_size
    patch_width = patch_size[1] if isinstance(patch_size, tuple) else patch_size
    grid_size = (input_tensor.shape[-2] // patch_height, input_tensor.shape[-1] // patch_width)

    probabilities = logits.softmax(dim=-1)[0]
    confidence, class_index = probabilities.max(dim=0)
    categories = weights.meta.get("categories", ())
    prediction = categories[class_index.item()] if categories else str(class_index.item())

    return AttentionRun(
        image=shown_image,
        attention=tuple(captured),
        grid_size=grid_size,
        prediction=prediction,
        confidence=confidence.item(),
    )


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _grid_image(
    image: Image.Image,
    grid_size: tuple[int, int],
    selected_patch: tuple[int, int] | None = None,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    rows, columns = grid_size
    patch_width = result.width / columns
    patch_height = result.height / rows

    for column in range(columns + 1):
        x = round(column * patch_width)
        draw.line((x, 0, x, result.height), fill="black", width=1)
    for row in range(rows + 1):
        y = round(row * patch_height)
        draw.line((0, y, result.width, y), fill="black", width=1)

    if selected_patch is not None:
        row, column = selected_patch
        box = (
            round(column * patch_width),
            round(row * patch_height),
            round((column + 1) * patch_width),
            round((row + 1) * patch_height),
        )
        draw.rectangle(box, outline="#00e5ff", width=max(2, result.width // 112))
    return result


def _attention_image(
    run: AttentionRun,
    layer: int,
    head: int | None,
    selected_patch: tuple[int, int],
    opacity: float,
    render_mode: str = "grayscale",
) -> Image.Image:
    rows, columns = run.grid_size
    row, column = selected_patch
    query_token = 1 + row * columns + column
    layer_attention = run.attention[layer][0]
    scores = layer_attention[:, query_token, 1:]
    if head is None:
        scores = scores.mean(dim=0)
    else:
        scores = scores[head]

    heatmap = scores.reshape(rows, columns).numpy()
    low, high = np.percentile(heatmap, (5, 95))
    heatmap = np.clip((heatmap - low) / max(high - low, 1e-12), 0, 1)
    if render_mode not in HEATMAP_RENDER_OPTIONS.values():
        raise ValueError(f"Unknown heatmap render mode: {render_mode}")

    result = (
        run.image.convert("RGB")
        if render_mode == "color"
        else ImageOps.grayscale(run.image).convert("RGB")
    )
    strength = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    strength = strength.resize(run.image.size, Image.Resampling.NEAREST)
    alpha = np.asarray(strength, dtype=np.float32)[..., None] / 255 * opacity * 0.75
    base = np.asarray(result, dtype=np.float32)
    yellow = np.array([255, 220, 0], dtype=np.float32)
    pixels = base + alpha * (yellow - base)
    result = Image.fromarray(pixels.round().astype(np.uint8), mode="RGB")

    draw = ImageDraw.Draw(result)
    patch_width = result.width / columns
    patch_height = result.height / rows
    for grid_row in range(rows + 1):
        y = round(grid_row * patch_height)
        draw.line((0, y, result.width, y), fill=(0, 0, 0, 150), width=1)
    for grid_column in range(columns + 1):
        x = round(grid_column * patch_width)
        draw.line((x, 0, x, result.height), fill=(0, 0, 0, 150), width=1)

    marker_width = max(1, result.width // 224)
    marker_length = max(3, round(min(patch_width, patch_height) * 0.28))
    for patch_row in range(rows):
        for patch_column in range(columns):
            value = float(heatmap[patch_row, patch_column])
            if value < 0.5:
                blend = value * 2
                start = np.array([90, 90, 90])
                end = np.array([255, 140, 0])
            else:
                blend = (value - 0.5) * 2
                start = np.array([255, 140, 0])
                end = np.array([255, 245, 0])
            marker_color = tuple(np.round(start + blend * (end - start)).astype(int))

            left = round(patch_column * patch_width) + 1
            top = round(patch_row * patch_height) + 1
            right = round((patch_column + 1) * patch_width) - 1
            bottom = round((patch_row + 1) * patch_height) - 1
            corners = (
                ((left + marker_length, top), (left, top), (left, top + marker_length)),
                ((right - marker_length, top), (right, top), (right, top + marker_length)),
                ((left, bottom - marker_length), (left, bottom), (left + marker_length, bottom)),
                ((right, bottom - marker_length), (right, bottom), (right - marker_length, bottom)),
            )
            for corner in corners:
                draw.line(corner, fill=marker_color, width=marker_width, joint="curve")

    draw.rectangle(
        (
            round(column * patch_width),
            round(row * patch_height),
            round((column + 1) * patch_width),
            round((row + 1) * patch_height),
        ),
        outline=(0, 229, 255, 255),
        width=max(2, result.width // 112),
    )
    return result.convert("RGB")


class ViTAttentionDemo:
    """A Jupyter widget for loading a ViT and exploring patch attention."""

    display_size = 448

    def __init__(self, device: torch.device | str | None = None):
        self.device = torch.device(device) if device is not None else _best_device()
        self.model = None
        self.weights = None
        self.run: AttentionRun | None = None
        self.source_image = Image.open(DEFAULT_IMAGE).convert("RGB")
        self.selected_patch: tuple[int, int] | None = None

        self.model_picker = widgets.Dropdown(
            options=MODEL_OPTIONS,
            description="Model",
            layout=widgets.Layout(width="310px"),
        )
        self.example_picker = widgets.Dropdown(
            options=IMAGENET_EXAMPLES,
            description="ImageNet example",
            layout=widgets.Layout(width="310px"),
        )
        self.load_button = widgets.Button(description="Load model", button_style="primary")
        self.upload = widgets.FileUpload(accept="image/*", multiple=False, description="Upload image")
        self.layer = widgets.IntSlider(description="Layer", min=1, max=1, value=1, disabled=True)
        self.head = widgets.Dropdown(description="Head", options=[("Mean", None)], disabled=True)
        self.opacity = widgets.FloatSlider(
            description="Overlay", min=0.1, max=1.0, step=0.05, value=0.7, disabled=True
        )
        self.render_mode = widgets.Dropdown(
            description="Render",
            options=HEATMAP_RENDER_OPTIONS,
            value="grayscale",
            disabled=True,
        )
        self.status = widgets.HTML(
            f"<b>Starting ViT-B/16...</b> Running on {self.device}."
        )
        self.source_view = widgets.Image(
            format="png",
            width=self.display_size,
            height=self.display_size,
            layout=widgets.Layout(width=f"{self.display_size}px", height=f"{self.display_size}px"),
        )
        self.heatmap_view = widgets.Image(
            format="png",
            width=self.display_size,
            height=self.display_size,
            layout=widgets.Layout(width=f"{self.display_size}px", height=f"{self.display_size}px"),
        )
        self.source_event = Event(source=self.source_view, watched_events=["click"])
        self.source_event.on_dom_event(self._on_image_click)

        preview = ImageOps.fit(self.source_image, (self.display_size, self.display_size))
        self.source_view.value = _png_bytes(_grid_image(preview, (14, 14)))
        self.heatmap_view.value = _png_bytes(preview)

        self.load_button.on_click(self._load_model)
        self.example_picker.observe(self._on_example, names="value")
        self.upload.observe(self._on_upload, names="value")
        self.layer.observe(self._redraw, names="value")
        self.head.observe(self._redraw, names="value")
        self.opacity.observe(self._redraw, names="value")
        self.render_mode.observe(self._redraw, names="value")

        controls = widgets.VBox([
            widgets.HBox([self.model_picker, self.load_button]),
            widgets.HBox([self.example_picker, self.upload]),
            widgets.HBox([self.layer, self.head, self.opacity, self.render_mode]),
            self.status,
        ])
        images = widgets.HBox([
            widgets.VBox([widgets.HTML("<b>Click a query patch</b>"), self.source_view]),
            widgets.VBox([widgets.HTML("<b>Attention to other patches</b>"), self.heatmap_view]),
        ])
        self.widget = widgets.VBox([controls, images])

    def _load_model(self, _button=None) -> None:
        model_name = self.model_picker.value
        self.load_button.disabled = True
        self.status.value = f"<b>Loading {model_name}...</b>"

        def show_download_progress(downloaded: int, total: int | None) -> None:
            downloaded_mb = downloaded / (1024 * 1024)
            if total:
                percent = downloaded / total
                total_mb = total / (1024 * 1024)
                self.status.value = (
                    f"<b>Downloading model weights:</b> {downloaded_mb:.0f}/{total_mb:.0f} MB "
                    f"({percent:.0%})"
                )
            else:
                self.status.value = f"<b>Downloading model weights:</b> {downloaded_mb:.0f} MB"

        try:
            model, weights = load_vit_model(model_name, self.device, show_download_progress)
            self.model, self.weights = model, weights
            self.status.value = f"<b>{model_name} loaded.</b> Analyzing the example image..."
            if self.source_image is not None:
                self._run_image()
        except Exception as error:
            self.status.value = f"<b>Could not load model:</b> {error}"
        finally:
            self.load_button.disabled = False

    def _on_example(self, change) -> None:
        image_path = change["new"]
        if image_path is None:
            return
        self.source_image = Image.open(image_path).convert("RGB")
        if self.model is None:
            preview = ImageOps.fit(self.source_image, (self.display_size, self.display_size))
            self.source_view.value = _png_bytes(_grid_image(preview, (14, 14)))
            self.heatmap_view.value = _png_bytes(preview)
            self.status.value = "<b>Example ready.</b> Load a model to calculate attention."
        else:
            self._run_image()

    def _on_upload(self, change) -> None:
        files = change["new"]
        if not files:
            return
        uploaded = files[0] if isinstance(files, (tuple, list)) else next(iter(files.values()))
        content = uploaded["content"] if isinstance(uploaded, dict) else uploaded.content
        self.source_image = Image.open(io.BytesIO(bytes(content))).convert("RGB")
        self.example_picker.index = None
        if self.model is None:
            preview = self.source_image.copy()
            preview.thumbnail((self.display_size, self.display_size))
            self.source_view.value = _png_bytes(preview)
            self.status.value = "<b>Image ready.</b> Load a model to calculate attention."
        else:
            self._run_image()

    def _run_image(self) -> None:
        self.status.value = "<b>Running inference and collecting all attention layers...</b>"
        try:
            self.run = run_attention(self.model, self.weights, self.source_image, self.device)
            self.selected_patch = (self.run.grid_size[0] // 2, self.run.grid_size[1] // 2)
            heads = self.run.attention[0].shape[1]
            self.layer.max = len(self.run.attention)
            self.layer.value = 1
            self.layer.disabled = False
            self.head.options = [("Mean", None)] + [(str(index + 1), index) for index in range(heads)]
            self.head.disabled = False
            self.opacity.disabled = False
            self.render_mode.disabled = False
            self.status.value = (
                f"<b>Prediction:</b> {self.run.prediction} "
                f"({self.run.confidence:.1%}) &nbsp; "
                f"<b>Selected patch:</b> row {self.selected_patch[0] + 1}, "
                f"column {self.selected_patch[1] + 1}"
            )
            self._redraw()
        except Exception as error:
            self.status.value = f"<b>Could not analyze image:</b> {error}"

    def _on_image_click(self, event) -> None:
        if self.run is None:
            return
        rows, columns = self.run.grid_size
        width = max(float(event.get("boundingRectWidth", self.display_size)), 1)
        height = max(float(event.get("boundingRectHeight", self.display_size)), 1)
        column = min(int(event["relativeX"] / width * columns), columns - 1)
        row = min(int(event["relativeY"] / height * rows), rows - 1)
        self.selected_patch = (row, column)
        self.status.value = (
            f"<b>Prediction:</b> {self.run.prediction} ({self.run.confidence:.1%}) &nbsp; "
            f"<b>Selected patch:</b> row {row + 1}, column {column + 1}"
        )
        self._redraw()

    def _redraw(self, _change=None) -> None:
        if self.run is None or self.selected_patch is None:
            return
        source = _grid_image(self.run.image, self.run.grid_size, self.selected_patch)
        heatmap = _attention_image(
            self.run,
            layer=self.layer.value - 1,
            head=self.head.value,
            selected_patch=self.selected_patch,
            opacity=self.opacity.value,
            render_mode=self.render_mode.value,
        )
        self.source_view.value = _png_bytes(source)
        self.heatmap_view.value = _png_bytes(heatmap)

    def show(self) -> None:
        display(self.widget)


def vit_attention_demo(
    device: torch.device | str | None = None,
    auto_load: bool = True,
) -> ViTAttentionDemo:
    """Create, display, and return an interactive ViT attention demo."""

    demo = ViTAttentionDemo(device=device)
    demo.show()
    if auto_load:
        demo._load_model()
    return demo
