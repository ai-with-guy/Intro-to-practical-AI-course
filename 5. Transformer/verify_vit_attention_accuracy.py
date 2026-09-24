"""Integration test for ViT attention capture using real Imagenette images.

This is intentionally a standalone developer test rather than a notebook cell.
It downloads Imagenette into a temporary directory by default, evaluates a
balanced subset, and deletes the data when it exits.

Run:
    uv run --extra cuda python verify_vit_attention_accuracy.py
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision.models import ViT_B_16_Weights, vit_b_16

from vit_attention_utils import _best_device, _capture_attention


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# Imagenette directory names are ImageNet WordNet IDs; values are ImageNet indices.
IMAGENETTE_CLASSES = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def download_imagenette(destination: Path) -> Path:
    """Download and extract Imagenette, returning its validation directory."""

    validation_dir = destination / "imagenette2-160" / "val"
    if validation_dir.exists():
        return validation_dir

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "imagenette2-160.tgz"
    print(f"Downloading Imagenette from {IMAGENETTE_URL}")
    with urllib.request.urlopen(IMAGENETTE_URL) as response, archive.open("wb") as output:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length else None
        with tqdm.wrapattr(
            response,
            "read",
            total=total,
            desc="Imagenette",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as source:
            shutil.copyfileobj(source, output)

    print("Extracting Imagenette...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")
    archive.unlink()
    return validation_dir


def balanced_samples(validation_dir: Path, samples_per_class: int) -> list[tuple[Path, int]]:
    samples = []
    for synset, class_index in IMAGENETTE_CLASSES.items():
        images = sorted((validation_dir / synset).glob("*.JPEG"))
        if len(images) < samples_per_class:
            raise RuntimeError(f"Only found {len(images)} images for {synset}.")
        samples.extend((path, class_index) for path in images[:samples_per_class])
    return samples


def load_batch(samples, transform, device):
    tensors = []
    labels = []
    for path, label in samples:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
        labels.append(label)
    return torch.stack(tensors).to(device), torch.tensor(labels, device=device)


@torch.inference_mode()
def evaluate_capture_safety(
    model,
    transform,
    samples,
    device,
    batch_size: int,
) -> tuple[float, float]:
    """Return normal and capture accuracies after checking model equivalence."""

    normal_correct = 0
    capture_correct = 0
    total = 0
    largest_logit_difference = 0.0

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        images, labels = load_batch(batch_samples, transform, device)

        logits_before = model(images)
        with _capture_attention(model) as captured:
            logits_during = model(images)
        logits_after = model(images)

        if any(weights is None for weights in captured):
            raise AssertionError("At least one encoder block did not return attention.")

        # need_weights=True can choose a different attention kernel, so permit tiny
        # floating-point differences while requiring predictions to be identical.
        torch.testing.assert_close(logits_during, logits_before, rtol=1e-3, atol=2e-4)
        torch.testing.assert_close(logits_after, logits_before, rtol=1e-5, atol=1e-6)
        if not torch.equal(logits_during.argmax(1), logits_before.argmax(1)):
            raise AssertionError("Attention capture changed one or more predictions.")

        largest_logit_difference = max(
            largest_logit_difference,
            (logits_during - logits_before).abs().max().item(),
        )
        normal_correct += (logits_before.argmax(1) == labels).sum().item()
        capture_correct += (logits_during.argmax(1) == labels).sum().item()
        total += labels.numel()
        print(f"Checked {total:>3}/{len(samples)} images", end="\r", flush=True)

    print()
    print(f"Largest capture logit difference: {largest_logit_difference:.2e}")
    return normal_correct / total, capture_correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Reuse/download Imagenette here instead of deleting it after the test.",
    )
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--minimum-accuracy", type=float, default=0.70)
    parser.add_argument("--device", help="For example: cpu, cuda, or xpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 1 or args.batch_size < 1:
        raise ValueError("Sample count and batch size must be positive.")

    temporary_directory = tempfile.TemporaryDirectory() if args.data_dir is None else nullcontext()
    with temporary_directory as temporary_path:
        data_dir = args.data_dir or Path(temporary_path)
        validation_dir = download_imagenette(data_dir)
        samples = balanced_samples(validation_dir, args.samples_per_class)

        device = torch.device(args.device) if args.device else _best_device()
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        print(f"Loading ViT-B/16 on {device}...")
        model = vit_b_16(weights=weights).to(device).eval()
        normal_accuracy, capture_accuracy = evaluate_capture_safety(
            model,
            weights.transforms(),
            samples,
            device,
            args.batch_size,
        )

        print(f"Normal top-1 accuracy:  {normal_accuracy:.1%}")
        print(f"Capture top-1 accuracy: {capture_accuracy:.1%}")
        if normal_accuracy < args.minimum_accuracy:
            raise AssertionError(
                f"Accuracy {normal_accuracy:.1%} is below the required "
                f"{args.minimum_accuracy:.1%}."
            )
        if capture_accuracy != normal_accuracy:
            raise AssertionError("Attention capture changed aggregate accuracy.")
        print("PASS: accuracy is healthy and attention capture is non-invasive.")


if __name__ == "__main__":
    main()
